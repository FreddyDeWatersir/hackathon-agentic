"""supply_agent.py — hybrid deterministic + LLM-advised procurement.

The deterministic engine is the source of truth: it does recipe math,
supplier scoring, budget enforcement, shelf-life caps, and the existing
cold-start/weekend/stockout boosts. The LLM is an ADVISOR that returns
coverage_days boosts (same units the deterministic layers already use).
The math layer then validates everything (shelf cap, budget, min order)
exactly as before.

A misbehaving LLM can only nudge targets within clamped ranges, never
break the math. If the LLM is unavailable or returns garbage, boosts
default to 0 and the deterministic flow runs exactly as before.

COVERAGE_DAYS LAYERS (applied in order, then capped by shelf life):
  1. Base: latency + gap + REORDER_BUFFER_DAYS + SAFETY_DAYS
  2. LLM global_boost: situational adjustment for ALL ingredients
  3. LLM ingredient_boosts[ing]: per-ingredient adjustment
  4. Cold start (day ≤ 5): +1.5
  5. Weekend pre-loading (Wed/Thu/Fri/Sat)
  6. Stockout feedback (yesterday's unavailable dishes): +1.0

Stateless aside from a per-day cache that prevents a duplicate LLM call
when the critic re-runs supply with an updated budget.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import litellm

from agents import memory as mem

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _weekday_index(obs: dict, day: int) -> int:
    """Day 1 is Monday (per contract); prefer the explicit string."""
    name = str(obs.get("day_of_week", "")).strip().lower()
    if name in _WEEKDAYS:
        return _WEEKDAYS[name]
    return (day - 1) % 7


@dataclass
class SupplyResult:
    actions: list[dict] = field(default_factory=list)
    requirements: dict[str, float] = field(default_factory=dict)
    spend: float = 0.0
    at_risk: list[str] = field(default_factory=list)
    expiring_soon: dict[str, float] = field(default_factory=dict)
    budget_left: float = 0.0


# --- Deterministic tunables -------------------------------------------------
REORDER_BUFFER_DAYS = 2.0
SAFETY_DAYS = 1.0
MAX_DAYS_ON_HAND = 7.0
DEFAULT_LEAD_DAYS = 2
DEFAULT_GAP_DAYS = 3.0
LATE_PENALTY = 0.20
LATENCY_PENALTY = 0.04
EXPIRY_WARN_DAYS = 2
LOW_COVER_DAYS = 2.0
COLD_START_DAYS = 5
COLD_START_BOOST = 1.5
STOCKOUT_FEEDBACK_BOOST = 1.0

# --- LLM advisory tunables --------------------------------------------------
LLM_GLOBAL_BOOST_MIN, LLM_GLOBAL_BOOST_MAX = -2.0, 3.0
LLM_PER_ING_BOOST_MIN, LLM_PER_ING_BOOST_MAX = -3.0, 5.0
MIN_COVERAGE_FLOOR = 0.5  # combined boosts can't zero out coverage

SUPPLY_LLM_SYSTEM = """\
You are the procurement strategist for a 22-table Italian restaurant in a
30-day survival game. A deterministic engine already handles recipe math,
supplier scoring, budget enforcement, and shelf-life caps. Your only job:
return coverage_days BOOSTS (additive, in days) that nudge the engine's
target stock up or down for situations the math cannot see.

WHEN TO NUDGE (each suggestion is roughly 0.5-1.5 days of boost):
- End-game: days_remaining <= 3 -> NEGATIVE global_boost to run inventory down.
- Weather: storm/rain on a high-traffic day -> modest negative boost.
- Weather: sunny weekend ahead -> positive boost anticipating higher demand.
- Alerts naming a supplier or ingredient -> boost alternatives BEFORE the hit.
- Demand trend: recent_5_days clearly accelerating -> positive global_boost.
- A specific dish repeatedly running out (across multiple days) -> positive
  ingredient_boost on that dish's ingredients.
- Reputation_band dropped to "Fair" or worse -> negative (demand will drop).

WHAT THE ENGINE ALREADY DOES (do NOT duplicate these):
- Cold-start buffer in first 5 days (+1.5 days automatically)
- Weekend pre-loading on Wed/Thu/Fri/Sat
- Stockout feedback on YESTERDAY's unavailable dishes (+1.0 day already applied)
- Supplier reliability scoring (penalises shortfalls and outages)

OUTPUT (only this JSON, no prose, no markdown fences):
{
  "global_boost": 0.0,
  "ingredient_boosts": {"Ingredient Name": 0.0},
  "reasoning": "one short sentence on what specifically triggered the boost"
}

CLAMPS: global_boost in [-2, +3], each ingredient_boost in [-3, +5].

DEFAULT to all zeros. The deterministic engine is well-tuned; you should
return zeros on most days. Only override when you see a specific signal
the engine can't (weather, alerts, trend, reputation, end-game)."""


# --- Per-day cache so the critic-triggered re-run skips a duplicate call ---
_LLM_ADVICE_CACHE: dict[int, dict] = {}


def _call_supply_llm(obs: dict, day: int, by_dish: dict[str, float],
                     requirements: dict[str, float]) -> dict:
    """Ask the LLM for coverage_days boosts. Returns {} if unavailable or
    on any error. Cached per day to avoid double-calls during critic re-runs."""
    global _LLM_ADVICE_CACHE
    # Reset cache at game start (same Python process may run multiple games).
    if day == 1:
        _LLM_ADVICE_CACHE = {}
    if day in _LLM_ADVICE_CACHE:
        return _LLM_ADVICE_CACHE[day]

    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")):
        _LLM_ADVICE_CACHE[day] = {}
        return {}

    ss = obs.get("service_summary", {}) or {}
    payload = {
        "day": day,
        "day_of_week": obs.get("day_of_week"),
        "days_remaining": obs.get("days_remaining"),
        "cash": obs.get("cash"),
        "weather_today": obs.get("weather_today"),
        "weather_forecast": obs.get("weather_forecast"),
        "alerts": obs.get("alerts"),
        "yesterday_covers": ss.get("total_covers"),
        "yesterday_walkout_band": ss.get("walkout_band"),
        "yesterday_dishes_unavailable_at": ss.get("dishes_unavailable_at") or {},
        "reputation_band": obs.get("reputation_band"),
        "customer_trend": obs.get("customer_trend"),
        "forecast_covers_today": round(sum(by_dish.values()), 1),
        "forecast_by_dish": {k: round(v, 1) for k, v in by_dish.items()},
        "active_ingredients": sorted(requirements.keys()),
        "dow_cover_averages": mem.dow_cover_averages(),
        "recent_5_days": [
            {"d": e["day"], "dow": e.get("covers_dow"), "cov": e["covers"]}
            for e in mem.get_recent(5)
        ],
    }

    try:
        resp = litellm.completion(
            model=MODEL, temperature=0.2, max_tokens=500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": SUPPLY_LLM_SYSTEM +
                 "\nOutput exactly valid JSON without trailing commas."},
                {"role": "user", "content": json.dumps(payload)},
            ])
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if txt.startswith("json"):
                txt = txt[4:].strip()
        txt = re.sub(r',\s*\}', '}', txt)
        txt = re.sub(r',\s*\]', ']', txt)
        result = json.loads(txt)
        if not isinstance(result, dict):
            result = {}
        _LLM_ADVICE_CACHE[day] = result
        return result
    except Exception as e:
        print(f"  [supply] LLM fallback: {e}")
        _LLM_ADVICE_CACHE[day] = {}
        return {}


def _parse_llm_boosts(advice: dict) -> tuple[float, dict[str, float]]:
    """Extract and clamp boosts from LLM advice. Returns (0.0, {}) on garbage."""
    try:
        global_boost = float(advice.get("global_boost", 0.0) or 0.0)
    except (TypeError, ValueError):
        global_boost = 0.0
    global_boost = max(LLM_GLOBAL_BOOST_MIN,
                       min(LLM_GLOBAL_BOOST_MAX, global_boost))

    per_ing_raw = advice.get("ingredient_boosts") or {}
    per_ing: dict[str, float] = {}
    if isinstance(per_ing_raw, dict):
        for k, v in per_ing_raw.items():
            if not isinstance(k, str):
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            per_ing[k] = max(LLM_PER_ING_BOOST_MIN,
                             min(LLM_PER_ING_BOOST_MAX, f))
    return global_boost, per_ing


def _daily_requirements(obs: dict, by_dish: dict[str, float]) -> dict[str, float]:
    """kg/day per ingredient = Σ forecast_units(dish) × recipe_kg."""
    req: dict[str, float] = {}
    for recipe in obs.get("menu_book", []) or []:
        dish = recipe.get("name")
        units = float(by_dish.get(dish, 0.0))
        if units <= 0:
            continue
        for comp in recipe.get("ingredients", []) or []:
            ing = comp.get("ingredient")
            kg = float(comp.get("quantity_kg", 0.0)) * units
            if ing:
                req[ing] = req.get(ing, 0.0) + kg
    return req


def _ingredients_of(obs: dict, dish_names: set[str]) -> set[str]:
    """Return the set of ingredient names that appear in the given dishes."""
    out: set[str] = set()
    for recipe in obs.get("menu_book", []) or []:
        if recipe.get("name") in dish_names:
            for comp in recipe.get("ingredients", []) or []:
                ing = comp.get("ingredient")
                if ing:
                    out.add(ing)
    return out


def _supplier_strikes(obs: dict) -> dict[str, int]:
    strikes: dict[str, int] = {}
    for d in obs.get("delivery_history", []) or []:
        sup = d.get("supplier")
        if not sup:
            continue
        short = d.get("delivered_kg", 0) + 1e-6 < d.get("ordered_kg", 0)
        if short or not d.get("on_time", True):
            strikes[sup] = strikes.get(sup, 0) + 1
    return strikes


def _alerted_suppliers(obs: dict) -> set[str]:
    bad: set[str] = set()
    alerts = " ".join(str(a) for a in obs.get("alerts", []) or []).lower()
    for sup in obs.get("supplier_catalog", []) or []:
        nm = sup.get("name", "")
        if nm and nm.lower() in alerts:
            bad.add(nm)
    return bad


def _days_until_delivery(sup: dict, today_wd: int) -> float:
    lead = int(sup.get("lead_time_days", DEFAULT_LEAD_DAYS))
    wd = sorted({_WEEKDAYS.get(str(d).strip().lower())
                 for d in sup.get("delivery_days", []) or []
                 if str(d).strip().lower() in _WEEKDAYS})
    if not wd:
        return float(lead) + DEFAULT_GAP_DAYS / 2.0
    for k in range(lead, lead + 14):
        if (today_wd + k) % 7 in wd:
            return float(k)
    return float(lead) + DEFAULT_GAP_DAYS


def _delivery_gap(sup: dict) -> float:
    """Worst gap between consecutive delivery days (days we must self-cover)."""
    wd = sorted({_WEEKDAYS.get(str(d).strip().lower())
                 for d in sup.get("delivery_days", []) or []
                 if str(d).strip().lower() in _WEEKDAYS})
    if len(wd) < 2:
        return DEFAULT_GAP_DAYS
    gaps = [((wd[(i + 1) % len(wd)] - wd[i]) % 7) or 7 for i in range(len(wd))]
    return float(max(gaps))


def propose(obs: dict, day: int, by_dish: dict[str, float],
            budget: float) -> SupplyResult:
    res = SupplyResult(budget_left=float(budget))
    today_wd = _weekday_index(obs, day)

    inv_kg = {i["ingredient"]: float(i.get("total_kg", 0.0))
              for i in obs.get("inventory", []) or []}

    # Shelf-life + batch-expiry maps. We use the MAX expires_in_days across
    # batches as a proxy for fresh-delivery shelf life (floored at 3 so we
    # never refuse to stock an ingredient entirely).
    shelf: dict[str, float] = {}
    expiry_by_batch: dict[str, list[tuple[float, float]]] = {}
    for i in obs.get("inventory", []) or []:
        ing = i["ingredient"]
        batches = i.get("batches", []) or []
        pairs = [(float(b.get("quantity_kg", 0.0)),
                  float(b.get("expires_in_days", 99) or 99))
                 for b in batches]
        expiry_by_batch[ing] = pairs
        max_exp = max((exp for _, exp in pairs), default=0.0)
        shelf[ing] = max(max_exp, 3.0)

    # Pending counts as incoming stock so we never double-order.
    pending: dict[str, float] = {}
    for po in obs.get("pending_orders", []) or []:
        ing = po.get("ingredient", "")
        pending[ing] = pending.get(ing, 0.0) + float(po.get("quantity_kg", 0.0))

    # Flag stock that will expire before we can plausibly use it.
    for i in obs.get("inventory", []) or []:
        ing = i["ingredient"]
        soon = sum(float(b.get("quantity_kg", 0.0))
                   for b in i.get("batches", []) or []
                   if b.get("expires_in_days", 99) <= EXPIRY_WARN_DAYS)
        if soon > 1e-6:
            res.expiring_soon[ing] = round(soon, 1)

    req = _daily_requirements(obs, by_dish)
    res.requirements = {k: round(v, 2) for k, v in req.items()}

    # Stockout feedback set: ingredients in dishes that ran out yesterday.
    ss_for_stockout = obs.get("service_summary", {}) or {}
    unavailable_dishes = set((ss_for_stockout.get("dishes_unavailable_at") or {}).keys())
    stockout_ingredients = _ingredients_of(obs, unavailable_dishes)

    # ---- LLM advisory (cached per-day) ---------------------------------
    # The LLM returns additive coverage_days boosts for situations the
    # deterministic layers can't see — weather, alerts, demand trends,
    # end-game, reputation drops. Output is clamped here, so a bad response
    # can only nudge targets a few days in either direction.
    llm_advice = _call_supply_llm(obs, day, by_dish, req)
    llm_global, llm_per_ing = _parse_llm_boosts(llm_advice)
    if (llm_global != 0.0 or llm_per_ing) and llm_advice.get("reasoning"):
        rounded_ing = {k: round(v, 1) for k, v in llm_per_ing.items()}
        boost_str = f"global={llm_global:+.1f}"
        if rounded_ing:
            boost_str += f", per-ing={rounded_ing}"
        print(f"  [supply] LLM: {llm_advice['reasoning']} ({boost_str})")

    strikes = _supplier_strikes(obs)
    alerted = _alerted_suppliers(obs)

    # Best supplier per ingredient by EFFECTIVE cost (price × risk × latency).
    best: dict[str, tuple] = {}
    for sup in obs.get("supplier_catalog", []) or []:
        name = sup.get("name", "")
        if not name or name in alerted:
            continue
        latency = _days_until_delivery(sup, today_wd)
        penalty = (1.0 + LATE_PENALTY * strikes.get(name, 0)
                   + LATENCY_PENALTY * latency)
        for ing, price in (sup.get("ingredients", {}) or {}).items():
            if ing not in req:
                continue
            eff = float(price) * penalty
            if ing not in best or eff < best[ing][0]:
                best[ing] = (eff, name, float(price),
                             float(sup.get("min_order_kg", 0.0)),
                             latency, _delivery_gap(sup))

    # Order most-urgent (lowest days-of-cover) ingredients first.
    ranked = []
    for ing, rate in req.items():
        on_hand = inv_kg.get(ing, 0.0) + pending.get(ing, 0.0)
        cover = on_hand / rate if rate > 1e-6 else 999.0
        ranked.append((cover, ing))
    ranked.sort()

    for cover, ing in ranked:
        entry = best.get(ing)
        if entry is None:
            res.at_risk.append(ing)
            continue
        _eff, supplier, price, min_order, latency, gap = entry
        rate = req[ing]
        on_hand = inv_kg.get(ing, 0.0) + pending.get(ing, 0.0)

        coverage_days = latency + gap + REORDER_BUFFER_DAYS + SAFETY_DAYS

        # LLM strategic adjustment (situations the math can't see).
        coverage_days += llm_global
        coverage_days += llm_per_ing.get(ing, 0.0)

        # Cold start: first week needs extra buffer (no DOW history yet).
        if day <= COLD_START_DAYS:
            coverage_days += COLD_START_BOOST

        # Weekend pre-loading around the Sunday closure.
        if today_wd == 4:    # Friday → arrives Mon
            coverage_days += 1.5
        elif today_wd == 3:  # Thursday → last chance to fatten Sat (arrives Fri)
            coverage_days += 1.5
        elif today_wd == 2:  # Wednesday → also pre-stages Sat
            coverage_days += 0.5
        elif today_wd == 5:  # Saturday → pre-stage Mon-Tue (arrives Mon)
            coverage_days += 0.5

        # Stockout feedback: yesterday's missing dishes signal under-forecast.
        if ing in stockout_ingredients:
            coverage_days += STOCKOUT_FEEDBACK_BOOST

        # Safety floor: never let combined boosts zero out coverage entirely.
        coverage_days = max(MIN_COVERAGE_FLOOR, coverage_days)

        cap = rate * min(shelf.get(ing, 7.0), MAX_DAYS_ON_HAND)
        target = min(rate * coverage_days, cap)

        # Stock that expires before delivery cannot be counted toward on-hand.
        expiring_before_delivery = sum(
            qty_b for qty_b, exp in expiry_by_batch.get(ing, [])
            if exp < latency
        )
        effective_on_hand = max(0.0, on_hand - expiring_before_delivery)

        qty = target - effective_on_hand
        if qty <= 0:
            continue
        if qty < min_order:
            on_hand_at_delivery = max(0.0, effective_on_hand - rate * latency)
            if min_order <= cap - on_hand_at_delivery:
                qty = min_order
            elif cover < latency + gap:
                qty = min_order
            else:
                continue

        cost = qty * price
        if cost > res.budget_left:
            res.at_risk.append(ing)
            continue
        res.actions.append({"tool": "place_order", "args": {
            "supplier": supplier, "ingredient": ing,
            "quantity_kg": round(qty, 1)}})
        res.budget_left -= cost
        res.spend += cost
        if cover < LOW_COVER_DAYS:
            res.at_risk.append(ing)

    res.spend = round(res.spend, 2)
    res.budget_left = round(res.budget_left, 2)
    return res