"""coordinator.py — LLM orchestrator + the strategy() entrypoint.

Pipeline each day:
  1. demand_agent (LLM)        → actions + forecast (covers, by_dish)
  2. coordinator LLM           → cash arbitration: how much supply may
                                 spend, final marketing, staffing risk bias
  3. supply_agent (det.)       → orders, constrained by that budget
  4. operations_agent (det.)   → staff level, using the forecast + bias
  5. merge actions, coordinator writes the single save_notes blob

The coordinator LLM only decides money/risk posture — never domain
actions — so a bad LLM response degrades to a safe reserve-based rule
instead of breaking the game. Run: `python -m agents.coordinator`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import litellm

from agents.runner import run_game
from agents import demand_agent, supply_agent, operations_agent

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")

# Deterministic fallback budget rule.
OVERHEAD_DAYS_RESERVE = 6
MIN_RESERVE = 2500.0

COORD_SYSTEM = """\
You are the financial controller of a restaurant in a 30-day survival
game. Going bankrupt (cash < 0) is a catastrophic -100,000. Daily fixed
+ staff overhead is large and unavoidable. You do NOT choose menu,
orders, or staff — you only set the money envelope and risk posture.

Inputs: cash, days_remaining, yesterday P&L, the demand forecast, what
supply roughly needs, and marketing the demand manager requested.

Respond with ONLY this JSON, no prose:
{
  "supply_budget": 0,            // EUR supply may spend this turn
  "marketing_spend": 0,          // 0-500, final approved amount
  "staff_bias": "lean|normal|safe",
  "reasoning": "one short sentence"
}
Keep enough cash to cover several days of overhead. Late game, you may
spend down reserves. Early game or low cash, be conservative."""


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
    reserve = max(MIN_RESERVE, OVERHEAD_DAYS_RESERVE * daily_overhead)
    return BudgetDecision(
        supply_budget=max(0.0, cash - reserve),
        marketing_spend=0.0 if cash < reserve else min(500.0, requested_mkt),
        staff_bias="normal",
        reasoning="deterministic reserve rule",
    )


def _coordinator_llm(obs: dict, demand, est_supply_need: float) -> BudgetDecision:
    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
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
    }
    try:
        resp = litellm.completion(
            model=MODEL, temperature=0.2, max_tokens=400,
            messages=[{"role": "system", "content": COORD_SYSTEM},
                      {"role": "user", "content": json.dumps(payload)}])
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if txt.startswith("json"):
                txt = txt[4:].strip()
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
    actions: list[dict] = []

    # 1. Demand (LLM) → forecast everything hangs off.
    demand = demand_agent.propose(observation, day)
    actions += demand.actions

    # 2. Coordinator (LLM) → money envelope + risk posture.
    est_need = _rough_supply_estimate(observation, demand.by_dish)
    budget = _coordinator_llm(observation, demand, est_need)

    # 3. Supply (deterministic), bounded by the coordinator's budget.
    supply = supply_agent.propose(observation, day, demand.by_dish,
                                  budget.supply_budget)
    actions += supply.actions

    # 4. Operations (deterministic), sized to the forecast + bias.
    ops = operations_agent.propose(observation, day, demand.expected_covers,
                                   budget.staff_bias)
    actions += ops.actions

    # Coordinator owns the single marketing (cash) action.
    if budget.marketing_spend > 0:
        actions.append({"tool": "set_marketing_spend",
                        "args": {"amount": round(budget.marketing_spend, 2)}})

    # 5. Coordinator owns the single notes store.
    notes = {
        "day": day,
        "cash": observation.get("cash"),
        "forecast_covers": round(demand.expected_covers, 1),
        "demand_conf": demand.confidence,
        "supply_spend": supply.spend,
        "supply_at_risk": supply.at_risk,
        "expiring": supply.expiring_soon,
        "staff": ops.staff_level,
        "service_risk": ops.service_risk,
        "budget": {"supply": round(budget.supply_budget, 2),
                   "mkt": round(budget.marketing_spend, 2),
                   "bias": budget.staff_bias},
    }
    actions.append({"tool": "save_notes",
                    "args": {"text": json.dumps(notes)[:4000]}})
    return actions


if __name__ == "__main__":
    result = run_game(strategy, team_name="multiagent", seed=42)