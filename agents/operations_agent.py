"""operations_agent.py — LLM-driven service / staffing specialist.

Serves the coordinator. Owns ONLY set_staff_level.

Staff is the single biggest controllable cost (120 EUR/person/day; 8 staff
= 960/day). The agent uses an LLM to look at expected covers, coordinator bias, 
and yesterday's service signals to decide exactly how many staff to schedule.
Falls back to a deterministic calculation if the LLM is unavailable.

CLOSED-DAY FAST PATH: when the coordinator signals a known closed day
(expected_covers=0, staff_bias="lean"), this agent drops directly to
MIN_STAFF without invoking the LLM and without applying MAX_STEP. The step
limit is anti-thrash for noisy demand — it's actively harmful on a day with
zero demand, where it forces us to pay for 5-6 idle staff (~720 EUR/day).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import litellm

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")

MIN_STAFF, MAX_STAFF = 3, 15

# Covers one staff member can comfortably handle per day. Adjusted by the
# coordinator's risk bias. ~20 → 90 covers ≈ 5 staff.
COVERS_PER_STAFF = {"lean": 24.0, "normal": 20.0, "safe": 16.0}

MAX_STEP = 2          # max staff change per day (anti-thrash on open days)
PEAK_WAIT_BAD = 10.0  # minutes; above this we were understaffed

SYSTEM_PROMPT = """\
You are the operations manager of a 22-table Italian restaurant in a 30-day
survival game. You own ONLY the staff level (from 3 to 15).

Hard rules:
- Staff is a major fixed cost (usually 120 EUR/day per person). Too many hurts profits.
- Too few causes walkouts, wait times, and ruins reputation.
- The coordinator provides a 'staff_bias' (lean, normal, safe) and an 'expected_covers' forecast.
- Adjust staff using yesterday's service signals (walkouts, wait times). 
- Avoid thrashing (don't change staff by more than 2-3 people at a time).

Respond with ONLY this valid JSON object, no prose:
{
  "actions": [
    {"tool": "set_staff_level", "args": {"level": 8}}
  ],
  "expected_cost": 960,
  "service_risk": "low|med|high",
  "reasoning": "one short sentence"
}
"""

@dataclass
class OperationsResult:
    actions: list[dict] = field(default_factory=list)
    staff_level: int = 0
    expected_cost: float = 0.0
    service_risk: str = "low"      # low | med | high
    reasoning: str = ""


def _closed_day_result(obs: dict) -> OperationsResult:
    """Drop to MIN_STAFF immediately on known closed days. Bypasses MAX_STEP
    because the step limit is meant to prevent thrash on noisy demand — on a
    day with zero demand it just forces wasted spend on idle staff."""
    cur = int(obs.get("staff_level", 8))
    cost_per = float(obs.get("staff_cost_per_person", 120.0))
    target = MIN_STAFF
    actions: list[dict] = []
    if target != cur:
        actions.append({"tool": "set_staff_level", "args": {"level": target}})
    return OperationsResult(
        actions=actions,
        staff_level=target,
        expected_cost=target * cost_per,
        service_risk="low",
        reasoning="closed day: minimum staffing (bypassing MAX_STEP)",
    )


def _fallback_propose(obs: dict, day: int, expected_covers: float,
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
    yesterday_covers = float(ss.get("total_covers") or 0)

    risk = "low"
    reason = f"forecast {expected_covers:.0f} covers → target {target}"

    # Asymmetric correction: understaffing signals override the target up —
    # BUT only when yesterday actually had customers. A "Many" walkout band
    # on a 0-cover day is a closed/stock-out artifact, not understaffing.
    real_understaff = (
        yesterday_covers > 0
        and (walkout in ("Some", "Many") or peak_wait > PEAK_WAIT_BAD or bottleneck)
    )
    if real_understaff:
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


def _call_llm(system: str, user: str) -> dict | None:
    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")):
        return None
    try:
        resp = litellm.completion(
            model=MODEL, temperature=0.2, max_tokens=400,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system + "\nOutput exactly valid JSON without trailing commas."},
                      {"role": "user", "content": user}])
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if txt.startswith("json"):
                txt = txt[4:].strip()
                
        txt = re.sub(r',\s*\}', '}', txt)
        txt = re.sub(r',\s*\]', ']', txt)

        return json.loads(txt)
    except Exception as e:
        print(f"  [operations] LLM fallback: {e}")
        return None


def propose(obs: dict, day: int, expected_covers: float,
            staff_bias: str = "normal") -> OperationsResult:

    # ----- Closed-day fast path -----------------------------------------
    # The coordinator routes known closed days here with expected_covers=0
    # and bias="lean". Skip the LLM entirely and bypass MAX_STEP — without
    # this, MAX_STEP=2 prevents us from dropping below cur-2 (e.g. 8→6),
    # which leaves us paying ~720 EUR/day for staff on a day with no service.
    if expected_covers <= 0 and staff_bias == "lean":
        return _closed_day_result(obs)

    ss = obs.get("service_summary", {}) or {}
    
    payload = {
        "day": day,
        "current_staff_level": int(obs.get("staff_level", 8)),
        "staff_cost_per_person": float(obs.get("staff_cost_per_person", 120.0)),
        "yesterday_walkout_band": ss.get("walkout_band"),
        "yesterday_covers": ss.get("total_covers"),
        "yesterday_peak_wait_minutes": ss.get("peak_wait_minutes"),
        "yesterday_kitchen_bottleneck": ss.get("kitchen_bottleneck_hours"),
        "table_utilization_peak": ss.get("table_utilization_peak"),
        "forecasted_expected_covers": expected_covers,
        "assigned_staff_bias": staff_bias,
        # Hint: walkouts on a 0-cover day reflect closure/stock-out, not staffing.
        "note": ("If yesterday_covers == 0, the walkout_band is a closed-day "
                 "or stock-out artifact and must NOT drive staff escalation."),
    }
    
    data = _call_llm(SYSTEM_PROMPT, json.dumps(payload))
    
    if not isinstance(data, dict):
        return _fallback_propose(obs, day, expected_covers, staff_bias)
        
    actions = data.get("actions", [])
    if not isinstance(actions, list):
        actions = []
        
    # Find the target staff level from actions
    target = int(obs.get("staff_level", 8))
    for a in actions:
        if a.get("tool") == "set_staff_level":
            target = int(a.get("args", {}).get("level", target))
            
    # Step-limit + clamp (Safety constraints even when using LLM)
    cur = int(obs.get("staff_level", 8))
    target = max(cur - MAX_STEP, min(cur + MAX_STEP, target))
    target = max(MIN_STAFF, min(MAX_STAFF, int(target)))
    
    # Overwrite the action with the clamped safe target
    safe_actions = []
    if target != cur or day == 1:
        safe_actions.append({"tool": "set_staff_level", "args": {"level": target}})
        
    return OperationsResult(
        actions=safe_actions,
        staff_level=target,
        expected_cost=float(data.get("expected_cost", target * float(obs.get("staff_cost_per_person", 120.0)))),
        service_risk=str(data.get("service_risk", "low")),
        reasoning=str(data.get("reasoning", "LLM decision"))
    )