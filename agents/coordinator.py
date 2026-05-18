"""coordinator.py — LLM orchestrator + the strategy() entrypoint.

Pipeline each day:
  1. demand_agent (LLM)        → actions + forecast (covers, by_dish)
  2. coordinator LLM           → cash arbitration: how much supply may
                                 spend, final marketing, staffing risk bias
  3. supply_agent (det.)       → orders, constrained by that budget
  4. operations_agent (det.)   → staff level, using the forecast + bias
  5. critic_agent (LLM)        → audits combined plan; may veto tools or
                                 override budget / staff_bias
  6. merge actions, memory records turn, coordinator writes save_notes

The coordinator LLM only decides money/risk posture — never domain
actions — so a bad LLM response degrades to a safe reserve-based rule
instead of breaking the game. Run: `python -m agents.coordinator`.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass

import litellm

from agents.runner import run_game
from agents import demand_agent, supply_agent, operations_agent, critic_agent
from agents import memory as mem

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")

# Deterministic fallback budget rule.
# 3 days is enough runway — 6 was too conservative and starved supply mid-game.
OVERHEAD_DAYS_RESERVE = 3
MIN_RESERVE = 1200.0


def _safe_price_ceiling(base_price: float) -> float:
    """Largest 2-decimal price strictly within the 0.8x-1.2x validator bound.

    The naive `round(base * 1.2, 2)` overshoots when `base * 1.2` lacks an
    exact float representation: e.g. `1.2 * 24.0 == 28.799999999999997`, and
    `round(..., 2) == 28.8`, which the API then rejects as 28.8 > 28.7999...
    `math.floor` to 2 decimals always lands strictly below the ceiling.
    """
    return math.floor(base_price * 1.2 * 100) / 100


COORD_SYSTEM = """\
You are the financial controller of a restaurant in a 30-day survival
game. Going bankrupt (cash < 0) is a catastrophic -100,000. Daily fixed
+ staff overhead is large and unavoidable. You do NOT choose menu,
orders, or staff — you only set the money envelope and risk posture.

Inputs: cash, days_remaining, yesterday P&L, the demand forecast, what
supply roughly needs, marketing the demand manager requested, historical
day-of-week cover averages (dow_avg), recent P&L trend, and happy-hour
consecutive streak.

Respond with ONLY this JSON, no prose:
{
  "supply_budget": 0,            // EUR supply may spend this turn
  "marketing_spend": 0,          // 0-500, final approved amount
  "staff_bias": "lean|normal|safe",
  "reasoning": "one short sentence"
}
Use dow_avg: if today is historically a high-demand day, prefer "safe";
if historically quiet, "lean" is fine. Keep enough cash to cover several
days of overhead. Late game, spend down reserves carefully."""


@dataclass
class BudgetDecision:
    supply_budget: float
    marketing_spend: float
    staff_bias: str
    reasoning: str


def _fallback_budget(obs: dict, requested_mkt: float) -> BudgetDecision:
    cash = float(obs.get("cash", 0.0))
    staff = float(obs.get("staff_level", 8))
    daily_overhead = staff * float(obs.get("staff_cost_per_person", 120.0)) + 300.0
    days_left = int(obs.get("days_remaining", 30))
    reserve_days = min(days_left, OVERHEAD_DAYS_RESERVE)
    reserve = max(MIN_RESERVE, reserve_days * daily_overhead)
    return BudgetDecision(
        supply_budget=max(0.0, cash - reserve),
        marketing_spend=0.0 if cash < reserve else min(500.0, requested_mkt),
        staff_bias="normal",
        reasoning="deterministic reserve rule",
    )


def _coordinator_llm(obs: dict, demand, est_supply_need: float) -> BudgetDecision:
    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")):
        return _fallback_budget(obs, demand.requested_marketing)
    payload = {
        "cash": obs.get("cash"),
        "days_remaining": obs.get("days_remaining"),
        "yesterday_revenue": obs.get("yesterday_revenue"),
        "yesterday_total_costs": obs.get("yesterday_total_costs"),
        "cost_breakdown": obs.get("cost_breakdown"),
        "reputation_band": obs.get("reputation_band"),
        "forecast_covers": demand.expected_covers,
        "estimated_supply_need_eur": round(est_supply_need, 2),
        "requested_marketing": demand.requested_marketing,
        "alerts": obs.get("alerts"),
        # Memory context
        "dow_cover_avg": mem.dow_cover_averages(),
        "today_dow": obs.get("day_of_week"),
        "happy_hour_streak": mem.happy_hour_streak(),
        "avg_pnl_5d": mem.recent_profit_trend(5),
        "recent_covers": [{"d": e["day"], "cov": e["covers"], "dow": e.get("covers_dow")}
                          for e in mem.get_recent(5)],
    }
    
    try:
        resp = litellm.completion(
            model=MODEL, temperature=0.2, max_tokens=400,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": COORD_SYSTEM + "\nOutput exactly valid JSON without trailing commas."},
                      {"role": "user", "content": json.dumps(payload)}])
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if txt.startswith("json"):
                txt = txt[4:].strip()
                
        txt = re.sub(r',\s*\}', '}', txt)
        txt = re.sub(r',\s*\]', ']', txt)

        d = json.loads(txt)
        cash = float(obs.get("cash", 0.0))
        return BudgetDecision(
            supply_budget=max(0.0, min(cash, float(d["supply_budget"]))),
            marketing_spend=max(0.0, min(500.0, float(d.get("marketing_spend", 0)))),
            staff_bias=str(d.get("staff_bias", "normal")),
            reasoning=str(d.get("reasoning", "")),
        )
    except Exception as e:
        print(f"  [coordinator] LLM fallback: {e}")
        return _fallback_budget(obs, demand.requested_marketing)


def _rough_supply_estimate(obs: dict, by_dish: dict) -> float:
    """Cheap pre-estimate of EUR supply needs, for the coordinator prompt."""
    cheapest = {}
    for s in obs.get("supplier_catalog", []) or []:
        for ing, price in (s.get("ingredients", {}) or {}).items():
            cheapest[ing] = min(cheapest.get(ing, 1e9), float(price))
    total = 0.0
    for m in obs.get("menu_book", []) or []:
        units = float(by_dish.get(m.get("name"), 0.0))
        for c in m.get("ingredients", []) or []:
            total += (units * float(c.get("quantity_kg", 0.0))
                      * cheapest.get(c.get("ingredient"), 0.0))
    return total * 2.0  # ~2 days of cover


def strategy(observation: dict, day: int) -> list[dict]:
    # Short-circuit for known closed days (e.g. Sunday).
    # Once we've observed a DOW with 0 covers at least once, skip all LLM
    # calls, supply orders, and marketing — just maintain minimum staff and
    # save notes to keep the memory chain intact.
    closed_dows = mem.known_closed_dows()
    if observation.get("day_of_week") in closed_dows:
        print(f"  [coordinator] day={day} ({observation.get('day_of_week')}) "
              f"is a known closed day — skipping orders/marketing.")
        ops = operations_agent.propose(observation, day, 0.0, "lean")
        mem.record_turn(
            obs=observation, day=day, forecast_covers=0.0,
            demand_conf="closed", budget_supply=0.0, budget_mkt=0.0,
            staff_bias="lean", staff_level=ops.staff_level,
            supply_spend=0.0, supply_at_risk=[], happy_hour=False,
        )
        notes_text = mem.build_notes(
            day=day, obs=observation, forecast_covers=0.0,
            demand_conf="closed", budget_supply=0.0, budget_mkt=0.0,
            staff_bias="lean", staff_level=ops.staff_level,
            supply_spend=0.0, supply_at_risk=[], happy_hour=False,
        )
        # Set all active-menu prices to the maximum allowed (1.2× base) to
        # suppress customer demand while the restaurant runs on skeleton crew.
        # The open-day path below resets them to base price the next morning.
        # Use _safe_price_ceiling, not round(base*1.2, 2): for bases like 24.0
        # the rounded value overshoots due to float precision and the API
        # rejects with "outside allowed range" (every Sunday hit this on
        # Grilled Salmon before the fix).
        price_suppress = []
        for dish in observation.get("menu_book", []) or []:
            if dish.get("is_active", False):
                max_p = _safe_price_ceiling(float(dish["base_price"]))
                if round(float(dish.get("current_price", 0.0)), 2) < max_p:
                    price_suppress.append({"tool": "set_price",
                                           "args": {"dish": dish["name"], "price": max_p}})
        return price_suppress + ops.actions + [{"tool": "save_notes", "args": {"text": notes_text}}]

    # If yesterday was a known closed day, prices were raised to suppress Sunday
    # demand. Reset every dish that drifted above base price so the demand agent
    # starts from a neutral baseline (its own set_price calls override these).
    _DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    today_idx = (_DOW_ORDER.index(observation.get("day_of_week", ""))
                 if observation.get("day_of_week") in _DOW_ORDER else -1)
    yesterday_dow = _DOW_ORDER[(today_idx - 1) % 7] if today_idx >= 0 else None
    price_reset_actions: list[dict] = []
    if yesterday_dow in closed_dows:
        for dish in observation.get("menu_book", []) or []:
            cur_p = round(float(dish.get("current_price", 0.0)), 2)
            base_p = round(float(dish.get("base_price", 0.0)), 2)
            if cur_p != base_p:
                price_reset_actions.append({"tool": "set_price",
                                            "args": {"dish": dish["name"], "price": base_p}})

    # 1. Demand (LLM) → forecast everything hangs off.
    demand = demand_agent.propose(observation, day)

    # 2. Coordinator (LLM) → money envelope + risk posture.
    est_need = _rough_supply_estimate(observation, demand.by_dish)
    budget = _coordinator_llm(observation, demand, est_need)

    # 3. Supply (deterministic), bounded by the coordinator's budget.
    supply = supply_agent.propose(observation, day, demand.by_dish,
                                  budget.supply_budget)

    # 4. Operations (deterministic), sized to the forecast + bias.
    ops = operations_agent.propose(observation, day, demand.expected_covers,
                                   budget.staff_bias)

    # Determine whether happy hour is in the proposed plan.
    happy_hour = any(a.get("tool") == "run_happy_hour" for a in demand.actions)

    # 5. Critic (LLM) → audit combined plan, may veto tools or override budget.
    assembled = demand.actions + supply.actions + ops.actions
    critique = critic_agent.audit(
        observation, day, assembled, budget,
        demand.expected_covers, happy_hour,
    )

    if critique.flags:
        print(f"  [critic] day={day}: {'; '.join(critique.flags)}")
    if critique.reasoning:
        print(f"  [critic] reasoning: {critique.reasoning}")

    # Apply critic overrides — re-run affected agents if budget or bias changed.
    budget_changed = (
        critique.override_supply_budget is not None
        or critique.override_marketing_spend is not None
        or critique.override_staff_bias is not None
    )
    if budget_changed:
        budget = BudgetDecision(
            supply_budget=(
                critique.override_supply_budget
                if critique.override_supply_budget is not None
                else budget.supply_budget
            ),
            marketing_spend=(
                critique.override_marketing_spend
                if critique.override_marketing_spend is not None
                else budget.marketing_spend
            ),
            staff_bias=(
                critique.override_staff_bias
                if critique.override_staff_bias is not None
                else budget.staff_bias
            ),
            reasoning=budget.reasoning,
        )
        supply = supply_agent.propose(observation, day, demand.by_dish,
                                      budget.supply_budget)
        ops = operations_agent.propose(observation, day, demand.expected_covers,
                                       budget.staff_bias)

    # Filter vetoed tools and assemble final action list.
    veto = set(critique.veto_tools)
    actions: list[dict] = [a for a in demand.actions if a.get("tool") not in veto]
    actions += supply.actions
    actions += ops.actions

    # Coordinator owns the single marketing action.
    if budget.marketing_spend > 0 and "set_marketing_spend" not in veto:
        actions.append({"tool": "set_marketing_spend",
                        "args": {"amount": round(budget.marketing_spend, 2)}})

    # 6. Record this turn in in-process memory.
    final_happy_hour = any(a.get("tool") == "run_happy_hour" for a in actions)
    mem.record_turn(
        obs=observation,
        day=day,
        forecast_covers=demand.expected_covers,
        demand_conf=demand.confidence,
        budget_supply=budget.supply_budget,
        budget_mkt=budget.marketing_spend,
        staff_bias=budget.staff_bias,
        staff_level=ops.staff_level,
        supply_spend=supply.spend,
        supply_at_risk=supply.at_risk,
        happy_hour=final_happy_hour,
    )

    # Persist memory blob (includes rolling 7-day history + dow averages).
    actions.append({"tool": "save_notes",
                    "args": {"text": mem.build_notes(
                        day=day,
                        obs=observation,
                        forecast_covers=demand.expected_covers,
                        demand_conf=demand.confidence,
                        budget_supply=budget.supply_budget,
                        budget_mkt=budget.marketing_spend,
                        staff_bias=budget.staff_bias,
                        staff_level=ops.staff_level,
                        supply_spend=supply.spend,
                        supply_at_risk=supply.at_risk,
                        happy_hour=final_happy_hour,
                        critique_flags=critique.flags,
                    )}})
    # Price resets go first so demand_agent's set_price calls can override per dish.
    return price_reset_actions + actions


if __name__ == "__main__":
    result = run_game(strategy, team_name="multitalyagent", seed=42)