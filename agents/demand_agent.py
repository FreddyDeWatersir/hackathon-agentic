"""demand_agent.py — LLM pricing / menu / promo strategist.

Two regression fixes:
  1. Deterministic guard: a set_menu with < 5 valid dishes is DROPPED
     (the current menu persists — safe). The LLM can no longer cause a
     "Menu needs at least 5 dishes" rejection.
  2. Prompt softened: the back-channel must NOT shrink the menu (a narrow
     menu suppresses demand per the strategy guide). The LLM keeps a
     broad menu and only uses pricing / daily_special to steer toward
     robust dishes; supply's deterministic reallocation does the real
     work of surviving the Sunday gap.

Fallback unchanged: no key / bad JSON ⇒ no menu/price change, safe.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import litellm

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")
MIN_MENU = 5

SYSTEM_PROMPT = """\
You are the pricing & menu manager of a 22-table Italian restaurant in a
30-day survival game.

A calibrated demand MODEL (not you) predicts demand — given under
MODEL_FORECAST, which also lists robust_dishes_today and
fragile_dishes_today (which dishes can/can't be kept fresh today; the
supplier calendar starves short-shelf ingredients on some days, notably
Sundays). Do NOT forecast.

Menu policy: KEEP A BROAD MENU. Never drop below 5 dishes; prefer
offering most or all dishes — a narrow menu suppresses demand. Do NOT
remove fragile dishes from the menu. Instead, on fragile-dish days:
- offer_daily_special on a ROBUST dish, and
- you MAY nudge fragile dishes toward the TOP of their 0.8x-1.2x band so
  demand naturally shifts to what the kitchen can serve.

Other levers: set_price within 0.8x-1.2x base (raise into 'surge',
discount on 'drop'/'recover', moderate moves only); run_happy_hour on
slow days; requested_marketing 0-500 (worth more when demand is soft).

Respond with ONLY this JSON, no prose, no markdown:
{
  "actions": [
    {"tool": "set_price", "args": {"dish": "...", "price": 0.0}},
    {"tool": "set_menu", "args": {"dishes": ["...", "...", "...", "...", "..."]}},
    {"tool": "run_happy_hour", "args": {}},
    {"tool": "offer_daily_special", "args": {"dish": "..."}}
  ],
  "requested_marketing": 0,
  "reasoning": "one short sentence"
}"""


@dataclass
class DemandResult:
    actions: list[dict] = field(default_factory=list)
    requested_marketing: float = 0.0
    reasoning: str = ""


def _base_prices(obs: dict) -> dict[str, float]:
    return {m.get("name"): float(m.get("base_price", 0.0))
            for m in obs.get("menu_book", []) or [] if m.get("name")}


def _valid_dishes(obs: dict) -> set:
    return {m.get("name") for m in obs.get("menu_book", []) or [] if m.get("name")}


def _sanitize(actions: list, obs: dict) -> list:
    """Clamp prices to band; drop any invalid (<5 valid) set_menu."""
    base = _base_prices(obs)
    valid = _valid_dishes(obs)
    out = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        tool = a.get("tool")
        if tool == "set_price":
            args = a.get("args", {}) or {}
            bp = base.get(args.get("dish"))
            try:
                p = float(args.get("price"))
            except (TypeError, ValueError):
                continue
            if bp:
                lo, hi = bp * 0.8 * 1.002, bp * 1.2 * 0.998
                args["price"] = round(max(lo, min(hi, p)), 2)
            out.append({"tool": "set_price", "args": args})
        elif tool == "set_menu":
            dishes = (a.get("args", {}) or {}).get("dishes", []) or []
            dishes = [d for d in dict.fromkeys(dishes) if d in valid]
            if len(dishes) >= MIN_MENU:           # else drop → menu persists
                out.append({"tool": "set_menu", "args": {"dishes": dishes}})
        else:
            out.append(a)
    return out


def _compact_observation(obs: dict) -> dict:
    inv_expiry = {
        i["ingredient"]: min((b.get("expires_in_days", 99)
                              for b in i.get("batches", []) or []), default=99)
        for i in obs.get("inventory", []) or []
    }
    return {
        "day": obs.get("day"),
        "day_of_week": obs.get("day_of_week"),
        "days_remaining": obs.get("days_remaining"),
        "reputation_band": obs.get("reputation_band"),
        "customer_trend": obs.get("customer_trend"),
        "weather_today": obs.get("weather_today"),
        "weather_forecast": obs.get("weather_forecast"),
        "active_menu": obs.get("active_menu"),
        "menu_book": [
            {"name": m.get("name"), "category": m.get("category"),
             "base_price": m.get("base_price"),
             "current_price": m.get("current_price")}
            for m in obs.get("menu_book", []) or []
        ],
        "ingredient_min_expiry_days": inv_expiry,
        "alerts": obs.get("alerts"),
    }


def _call_llm(system: str, user: str) -> dict | None:
    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        return None
    try:
        resp = litellm.completion(
            model=MODEL, temperature=0.3, max_tokens=1000,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if txt.startswith("json"):
                txt = txt[4:].strip()
        return json.loads(txt)
    except Exception as e:
        print(f"  [demand] LLM fallback: {e}")
        return None


def propose(obs: dict, day: int, mem_summary: dict,
            model_forecast: dict) -> DemandResult:
    payload = {"MODEL_FORECAST": model_forecast,
               "MEMORY": mem_summary or {},
               "OBSERVATION": _compact_observation(obs)}
    user = f"Day {day}/30.\n{json.dumps(payload, indent=2)}"
    data = _call_llm(SYSTEM_PROMPT, user)

    if not isinstance(data, dict):
        return DemandResult(reasoning="LLM unavailable — no menu/price change")

    actions = data.get("actions")
    if not isinstance(actions, list):
        actions = []
    actions = _sanitize(actions, obs)

    try:
        req_mkt = float(data.get("requested_marketing", 0.0) or 0.0)
    except (TypeError, ValueError):
        req_mkt = 0.0

    return DemandResult(actions=actions,
                        requested_marketing=max(0.0, min(500.0, req_mkt)),
                        reasoning=str(data.get("reasoning", "")))