"""operations_agent.py — deterministic service / staffing specialist (v2).

Serves the coordinator. Owns ONLY set_staff_level.

v2 fixes the bug that caused ~60% of the tourist_season bankruptcy:

  ROOT CAUSE (v1): when the kitchen ran out of food, covers went to 0 and
  walkouts went to "Many". v1 read "Many walkouts" as "understaffed" and
  ratcheted staff 5 → 13 into an EMPTY restaurant — ~€9,100 of wages
  burned serving nobody. Staffing up is the worst possible response to a
  supply failure.

  FIX 1 — no-service detection: if yesterday's covers were ≈0, the
  walkouts are a SUPPLY/menu failure, not a service one. Don't add staff;
  drop to minimum to conserve cash (there is nothing to serve anyway).

  FIX 2 — demand-justified ceiling: the walkout ratchet can NEVER push
  staff above what the forecast covers justify (+ a small slack). This
  structurally prevents the 5→13 balloon under any signal combination.

  Also: floor expected_covers by yesterday's actuals (same defensive move
  as supply), because the LLM forecast is unreliable.

Stateless: reads current staff_level from the observation each turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_STAFF, MAX_STAFF = 3, 15

# Covers one staff member handles per day, by coordinator risk bias.
COVERS_PER_STAFF = {"lean": 24.0, "normal": 20.0, "safe": 16.0}

MAX_STEP = 2          # max staff change per day (anti-thrash), increases only
PEAK_WAIT_BAD = 10.0  # minutes; above this we were genuinely understaffed
CEILING_SLACK = 3     # most staff allowed ABOVE the demand-justified target
NO_SERVICE_ABS = 5    # covers at/below this ⇒ effectively no service yesterday
NO_SERVICE_FRAC = 0.2 # ...or below this fraction of the forecast


@dataclass
class OperationsResult:
    actions: list[dict] = field(default_factory=list)
    staff_level: int = 0
    expected_cost: float = 0.0
    service_risk: str = "low"      # low | med | high
    reasoning: str = ""


def propose(obs: dict, day: int, expected_covers: float,
            staff_bias: str = "normal") -> OperationsResult:
    cur = int(obs.get("staff_level", 8))
    cost_per = float(obs.get("staff_cost_per_person", 120.0))
    ss = obs.get("service_summary", {}) or {}

    covers_yest = float(ss.get("total_covers") or 0.0)
    walkout = str(ss.get("walkout_band", "None"))
    peak_wait = float(ss.get("peak_wait_minutes", 0.0))
    bottleneck = bool(ss.get("kitchen_bottleneck_hours", []))
    util_peak = float(ss.get("table_utilization_peak", 0.0))

    # Defensive: don't trust an under-forecast — floor by yesterday's actuals.
    eff_covers = max(float(expected_covers), covers_yest)

    ratio = COVERS_PER_STAFF.get(staff_bias, COVERS_PER_STAFF["normal"])
    demand_target = max(MIN_STAFF, round(eff_covers / ratio + 0.5))
    ceiling = max(MIN_STAFF, demand_target + CEILING_SLACK)

    # Was yesterday effectively a no-service day? (Only meaningful after day 1.)
    no_service = day > 1 and covers_yest <= max(
        NO_SERVICE_ABS, NO_SERVICE_FRAC * float(expected_covers))

    if no_service:
        # Walkouts yesterday meant NO FOOD. Today, demand returns.
        # Staff for today's forecast, not yesterday's failure.
        target = demand_target
        risk = "med"
        reason = (f"covers≈{covers_yest:.0f}: supply/menu failure yesterday, "
                  f"recovering staff for forecast {expected_covers:.0f}")
    elif walkout in ("Some", "Many") or peak_wait > PEAK_WAIT_BAD or bottleneck:
        # Genuine load: we WERE serving and still hit limits.
        target = max(demand_target, cur + 1)
        risk = "high" if walkout == "Many" else "med"
        reason = (f"genuine load (covers={covers_yest:.0f}, "
                  f"walkout={walkout}, peak_wait={peak_wait:.0f})")
        target = min(cur + MAX_STEP, target)          # step-limit increases
    elif (walkout == "None" and peak_wait < 2.0 and util_peak < 0.5
          and cur > demand_target):
        target = cur - 1                              # clean & slack, trim
        risk = "low"
        reason = "clean & slack, trimming"
    else:
        target = demand_target
        risk = "low"
        reason = f"forecast {expected_covers:.0f} → target {demand_target}"

    # HARD demand-justified ceiling: nothing can balloon staff past this.
    target = min(target, ceiling)
    if not no_service:
        target = max(cur - MAX_STEP, target)          # step-limit decreases
    target = max(MIN_STAFF, min(MAX_STAFF, int(target)))

    res = OperationsResult(staff_level=target,
                           expected_cost=target * cost_per,
                           service_risk=risk, reasoning=reason)
    if target != cur or day == 1:
        res.actions.append({"tool": "set_staff_level",
                            "args": {"level": target}})
    return res