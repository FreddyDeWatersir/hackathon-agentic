"""supply_agent.py — deterministic procurement specialist.

Serves the coordinator. Owns ONLY place_order.

Given the demand agent's per-dish forecast, it computes the EXACT kg of
each ingredient needed (recipes carry quantity_kg), sizes orders to bridge
the real supply gap (lead time + delivery-day cadence), caps by shelf life
so it doesn't manufacture waste, picks the most reliable affordable
supplier, and never exceeds the budget the coordinator hands it.

Stateless: every signal it needs is in the observation each turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    requirements: dict[str, float] = field(default_factory=dict)   # kg/day
    spend: float = 0.0
    at_risk: list[str] = field(default_factory=list)               # couldn't cover
    expiring_soon: dict[str, float] = field(default_factory=dict)  # kg wasting
    budget_left: float = 0.0


# --- Tunables --------------------------------------------------------------
REORDER_BUFFER_DAYS = 2.0   # slack on top of the delivery gap
SAFETY_DAYS = 1.0           # extra cover for demand variance
MAX_DAYS_ON_HAND = 7.0      # hard ceiling on stock depth (waste guard)
DEFAULT_LEAD_DAYS = 2
DEFAULT_GAP_DAYS = 3.0
LATE_PENALTY = 0.20         # effective-cost bump per recent shortfall/late
LATENCY_PENALTY = 0.04      # effective-cost bump per day until delivery
EXPIRY_WARN_DAYS = 2        # flag stock expiring within this many days
LOW_COVER_DAYS = 2.0        # ingredient is "at risk" below this much cover


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
    shelf = {i["ingredient"]: float(i.get("shelf_life_days", 7))
             for i in obs.get("inventory", []) or []}

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
            res.at_risk.append(ing)  # nobody sells it / all in outage
            continue
        _eff, supplier, price, min_order, latency, gap = entry
        rate = req[ing]
        on_hand = inv_kg.get(ing, 0.0) + pending.get(ing, 0.0)

        coverage_days = latency + gap + REORDER_BUFFER_DAYS + SAFETY_DAYS
        cap = rate * min(shelf.get(ing, 7.0), MAX_DAYS_ON_HAND)
        target = min(rate * coverage_days, cap)

        qty = target - on_hand
        if qty <= 0:
            continue
        if qty < min_order:
            if min_order <= cap - on_hand:
                qty = min_order
            else:
                continue  # bumping to min_order would create waste

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