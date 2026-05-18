"""supply_agent.py — deterministic procurement specialist (v3).

v3 adds the only thing memory changes here: when memory reports the
demand regime is ramp/surge/recover, supply PRE-POSITIONS perishables —
it relaxes the shelf-life waste cap and treats every needed ingredient
as critical (always at least the supplier minimum), accepting some
spoilage because a stockout *during a surge* costs vastly more than
waste. In a stable/drop regime it behaves exactly as v2 (waste-tight).

Everything else (recipe-exact requirements, yesterday-sales floor,
delivery-aware sizing, visible at_risk) is unchanged from v2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _weekday_index(obs: dict, day: int) -> int:
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


REORDER_BUFFER_DAYS = 2.0
SAFETY_DAYS = 1.0
MAX_DAYS_ON_HAND = 7.0
CRITICAL_COVER_DAYS = 1.0
DEFAULT_LEAD_DAYS = 2
DEFAULT_GAP_DAYS = 3.0
LATE_PENALTY = 0.20
LATENCY_PENALTY = 0.04
EXPIRY_WARN_DAYS = 2
PREPOSITION_REGIMES = {"ramp", "surge", "recover"}
PREPOSITION_CAP_FACTOR = 2.0   # how far past the shelf-life cap we'll go


def _effective_by_dish(obs: dict, by_dish: dict[str, float]) -> dict[str, float]:
    valid = {m.get("name") for m in obs.get("menu_book", []) or []}
    eff = {k: float(v) for k, v in by_dish.items() if k in valid}
    sold = (obs.get("service_summary", {}) or {}).get("dishes_sold", {}) or {}
    for dish, n in sold.items():
        if dish in valid:
            eff[dish] = max(eff.get(dish, 0.0), float(n))
    return eff


def _daily_requirements(obs: dict, by_dish: dict[str, float]) -> dict[str, float]:
    req: dict[str, float] = {}
    for recipe in obs.get("menu_book", []) or []:
        units = float(by_dish.get(recipe.get("name"), 0.0))
        if units <= 0:
            continue
        for comp in recipe.get("ingredients", []) or []:
            ing = comp.get("ingredient")
            if ing:
                req[ing] = req.get(ing, 0.0) + float(
                    comp.get("quantity_kg", 0.0)) * units
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
    wd = sorted({_WEEKDAYS.get(str(d).strip().lower())
                 for d in sup.get("delivery_days", []) or []
                 if str(d).strip().lower() in _WEEKDAYS})
    if len(wd) < 2:
        return DEFAULT_GAP_DAYS
    gaps = [((wd[(i + 1) % len(wd)] - wd[i]) % 7) or 7 for i in range(len(wd))]
    return float(max(gaps))


def propose(obs: dict, day: int, by_dish: dict[str, float],
            budget: float, regime: str = "stable") -> SupplyResult:
    res = SupplyResult(budget_left=float(budget))
    today_wd = _weekday_index(obs, day)
    prepos = regime in PREPOSITION_REGIMES

    inv_kg = {i["ingredient"]: float(i.get("total_kg", 0.0))
              for i in obs.get("inventory", []) or []}
    shelf = {i["ingredient"]: float(i.get("shelf_life_days", 7))
             for i in obs.get("inventory", []) or []}

    pending: dict[str, float] = {}
    for po in obs.get("pending_orders", []) or []:
        ing = po.get("ingredient", "")
        pending[ing] = pending.get(ing, 0.0) + float(po.get("quantity_kg", 0.0))

    for i in obs.get("inventory", []) or []:
        ing = i["ingredient"]
        soon = sum(float(b.get("quantity_kg", 0.0))
                   for b in i.get("batches", []) or []
                   if b.get("expires_in_days", 99) <= EXPIRY_WARN_DAYS)
        if soon > 1e-6:
            res.expiring_soon[ing] = round(soon, 1)

    eff_by_dish = _effective_by_dish(obs, by_dish)
    req = _daily_requirements(obs, eff_by_dish)
    res.requirements = {k: round(v, 2) for k, v in req.items()}

    strikes = _supplier_strikes(obs)
    alerted = _alerted_suppliers(obs)

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

    ranked = sorted(
        ((inv_kg.get(ing, 0.0) + pending.get(ing, 0.0)) / rate
         if rate > 1e-6 else 999.0, ing)
        for ing, rate in req.items())

    for cover, ing in ranked:
        rate = req[ing]
        if rate <= 1e-6:
            continue
        entry = best.get(ing)
        if entry is None:
            res.at_risk.append(ing)
            continue
        _eff, supplier, price, min_order, latency, gap = entry
        on_hand = inv_kg.get(ing, 0.0) + pending.get(ing, 0.0)

        coverage_target = latency + gap + REORDER_BUFFER_DAYS + SAFETY_DAYS
        if on_hand >= rate * coverage_target:
            continue

        need = rate * coverage_target - on_hand
        # Pre-position during a surge: allow building past the shelf cap.
        cap_factor = PREPOSITION_CAP_FACTOR if prepos else 1.0
        usable_room = max(rate * min(shelf.get(ing, 7.0), MAX_DAYS_ON_HAND)
                          * cap_factor - on_hand, 0.0)
        qty = min(need, usable_room)

        # In a surge regime treat everything as critical (keep pipeline full).
        critical = prepos or cover < CRITICAL_COVER_DAYS
        if qty < min_order:
            if critical:
                qty = min_order
            elif usable_room >= min_order:
                qty = min_order
            else:
                res.at_risk.append(ing)
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

    res.spend = round(res.spend, 2)
    res.budget_left = round(res.budget_left, 2)
    return res