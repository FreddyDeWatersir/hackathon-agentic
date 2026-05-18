"""operations_agent.py — deterministic service / staffing specialist.

Serves the coordinator. Owns ONLY set_staff_level.

Staff is the single biggest controllable cost (120 EUR/person/day; 8 staff
= 960/day). The day-1 baseline ran 92 covers on 8 staff with zero wait and
0.23 table utilisation — heavily overstaffed. So the agent sizes staff to
the forecast covers, then hill-climbs against yesterday's service signals:
trim when service was clean and slack, add fast when waits / walkouts /
kitchen bottlenecks appear (reputation spirals are expensive, so it is
asymmetric — quick to add, slow to cut). Step-limited to avoid thrashing.

Stateless: reads current staff_level from the observation each turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_STAFF, MAX_STAFF = 3, 15

# Covers one staff member can comfortably handle per day. Adjusted by the
# coordinator's risk bias. ~20 → 90 covers ≈ 5 staff.
COVERS_PER_STAFF = {"lean": 24.0, "normal": 20.0, "safe": 16.0}

MAX_STEP = 2          # max staff change per day (anti-thrash)
PEAK_WAIT_BAD = 10.0  # minutes; above this we were understaffed


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

    ratio = COVERS_PER_STAFF.get(staff_bias, COVERS_PER_STAFF["normal"])
    target = max(MIN_STAFF, round(float(expected_covers) / ratio + 0.5))

    walkout = str(ss.get("walkout_band", "None"))
    peak_wait = float(ss.get("peak_wait_minutes", 0.0))
    bottleneck = bool(ss.get("kitchen_bottleneck_hours", []))
    util_peak = float(ss.get("table_utilization_peak", 0.0))

    risk = "low"
    reason = f"forecast {expected_covers:.0f} covers → target {target}"

    # Asymmetric correction: understaffing signals override the target up.
    if walkout in ("Some", "Many") or peak_wait > PEAK_WAIT_BAD or bottleneck:
        target = max(target, cur + 1)
        risk = "high" if walkout == "Many" else "med"
        reason += f"; understaffed (walkout={walkout}, peak_wait={peak_wait:.0f})"
    elif (walkout == "None" and peak_wait < 2.0 and util_peak < 0.5
          and cur > target):
        # Clean + slack: drift down toward target, but only one step.
        target = cur - 1
        reason += "; clean & slack, trimming"

    # Step-limit + clamp.
    target = max(cur - MAX_STEP, min(cur + MAX_STEP, target))
    target = max(MIN_STAFF, min(MAX_STAFF, int(target)))

    res = OperationsResult(staff_level=target,
                           expected_cost=target * cost_per,
                           service_risk=risk, reasoning=reason)
    if target != cur or day == 1:
        res.actions.append({"tool": "set_staff_level",
                            "args": {"level": target}})
    return res