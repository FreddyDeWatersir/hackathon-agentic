"""coordinator.py — LLM orchestrator + strategy() entrypoint (memory-aware).

Per day:
  1. rebuild Memory from observation["notes"]; update it with yesterday
  2. demand_agent (LLM)  — gets the memory summary → grounded forecast
  3. coordinator LLM     — cash posture, then a DETERMINISTIC FLOOR so it
                           can't starve supply while hoarding cash
  4. supply_agent (det.) — gets the regime → pre-positions before a surge
  5. operations_agent (det.)
  6. serialize Memory back into the single save_notes blob

Memory is rebuilt-from-notes every turn (never in-process), so this is
immune to the parallel-evaluate cross-game hazard. Run:
  RESTBENCH_DEBUG=1 python -m agents.coordinator --scenario tourist_season
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import litellm

from agents.runner import run_game
from agents import demand_agent, supply_agent, operations_agent
from agents.memory import Memory

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")
DEBUG = bool(os.getenv("RESTBENCH_DEBUG"))

OVERHEAD_DAYS_RESERVE = 6
MIN_RESERVE = 2500.0
HARD_RESERVE = 1500.0          # absolute floor we never spend supply below
SURGE_BUDGET_MULT = 1.6        # extra supply headroom when pre-positioning

COORD_SYSTEM = """\
You are the financial controller of a restaurant in a 30-day survival
game. Bankruptcy (cash<0) is a catastrophic -100,000. You set only the
money envelope and risk posture, not menu/orders/staff.

Respond with ONLY this JSON, no prose:
{"supply_budget":0,"marketing_spend":0,"staff_bias":"lean|normal|safe",
 "reasoning":"one short sentence"}
Withholding ingredient money while sitting on cash causes stockouts that
lose far more than the cash saved. Fund supply to its need unless cash
is genuinely tight."""


@dataclass
class BudgetDecision:
    supply_budget: float
    marketing_spend: float
    staff_bias: str
    reasoning: str


def _rough_supply_estimate(obs: dict, by_dish: dict) -> float:
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
    return total * 3.0  # ~3 days of cover


def _fallback_budget(obs: dict, mkt: float) -> BudgetDecision:
    cash = float(obs.get("cash", 0.0))
    staff = float(obs.get("staff_level", 8))
    oh = staff * float(obs.get("staff_cost_per_person", 120.0)) + 300.0
    reserve = max(MIN_RESERVE, OVERHEAD_DAYS_RESERVE * oh)
    return BudgetDecision(max(0.0, cash - reserve),
                          0.0 if cash < reserve else min(500.0, mkt),
                          "normal", "deterministic reserve rule")


def _coordinator_llm(obs: dict, demand) -> BudgetDecision:
    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        return _fallback_budget(obs, demand.requested_marketing)
    payload = {"cash": obs.get("cash"),
               "days_remaining": obs.get("days_remaining"),
               "yesterday_revenue": obs.get("yesterday_revenue"),
               "yesterday_total_costs": obs.get("yesterday_total_costs"),
               "reputation_band": obs.get("reputation_band"),
               "forecast_covers": demand.expected_covers,
               "requested_marketing": demand.requested_marketing}
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
            max(0.0, min(cash, float(d["supply_budget"]))),
            max(0.0, min(500.0, float(d.get("marketing_spend", 0)))),
            str(d.get("staff_bias", "normal")),
            str(d.get("reasoning", "")))
    except Exception as e:
        print(f"  [coordinator] LLM fallback: {e}")
        return _fallback_budget(obs, demand.requested_marketing)


def strategy(observation: dict, day: int) -> list[dict]:
    actions: list[dict] = []

    # 1. Memory: rebuild from notes (per-game, parallel-safe), update.
    mem = Memory.from_notes(observation.get("notes"))
    mem.update(observation, day)
    regime = mem.regime()

    # 2. Demand (LLM) with grounded memory.
    demand = demand_agent.propose(observation, day, mem.summary())
    actions += demand.actions
    mem.record_forecast(demand.expected_covers)

    # 3. Coordinator LLM posture + DETERMINISTIC FLOOR (kills the doom loop).
    budget = _coordinator_llm(observation, demand)
    cash = float(observation.get("cash", 0.0))
    est_need = _rough_supply_estimate(observation, demand.by_dish)
    if regime in ("ramp", "surge", "recover"):
        est_need *= SURGE_BUDGET_MULT
    floor = min(max(0.0, cash - HARD_RESERVE), est_need)
    supply_budget = max(budget.supply_budget, floor)

    # 4. Supply (det.) — regime-aware pre-positioning.
    supply = supply_agent.propose(observation, day, demand.by_dish,
                                  supply_budget, regime)
    actions += supply.actions

    # 5. Operations (det.).
    ops = operations_agent.propose(observation, day, demand.expected_covers,
                                   budget.staff_bias)
    actions += ops.actions

    if budget.marketing_spend > 0:
        actions.append({"tool": "set_marketing_spend",
                        "args": {"amount": round(budget.marketing_spend, 2)}})

    # 6. The single notes store IS the serialized memory (bounded).
    actions.append({"tool": "save_notes", "args": {"text": mem.to_notes()}})

    if DEBUG:
        print(f"\n===== day {day} cash={observation.get('cash')} "
              f"regime={regime} bias={mem.bias:.2f} =====")
        print(f"  DEMAND covers={demand.expected_covers:.0f} "
              f"conf={demand.confidence} by_dish={demand.by_dish}")
        print(f"  SUPPLY spend={supply.spend} budget={supply_budget:.0f} "
              f"(llm={budget.supply_budget:.0f} floor={floor:.0f}) "
              f"orders={[a['args'] for a in supply.actions]}")
        print(f"  at_risk={supply.at_risk} expiring={supply.expiring_soon}")
        print(f"  OPS staff={ops.staff_level} :: {ops.reasoning}")
        print(f"  mem.covers={mem.covers} peak={mem.peak} "
              f"starved={mem.starved_days} notes_chars={len(mem.to_notes())}")
    return actions


if __name__ == "__main__":
    result = run_game(strategy, team_name="multiagent", seed=42)