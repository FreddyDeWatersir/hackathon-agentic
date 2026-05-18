"""coordinator.py — LLM orchestrator + strategy() entrypoint.

Regression fix: the sourceability oracle is now fed memory's learned
INTRINSIC shelf life (mem.shelf), not transient inventory, so it no
longer collapses to "no robust dishes" once stock depletes. Everything
else (single-source-of-truth forecast, horizon-peak × band, #2 demand
reallocation onto robust dishes) is unchanged.

Run: RESTBENCH_DEBUG=1 python -m agents.coordinator --scenario tourist_season
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
HARD_RESERVE = 1500.0
SURGE_BUDGET_MULT = 1.6
SAFETY_BAND = 1.2

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
    return total * 3.0


def _fallback_budget(obs: dict, mkt: float) -> BudgetDecision:
    cash = float(obs.get("cash", 0.0))
    staff = float(obs.get("staff_level", 8))
    oh = staff * float(obs.get("staff_cost_per_person", 120.0)) + 300.0
    reserve = max(MIN_RESERVE, OVERHEAD_DAYS_RESERVE * oh)
    return BudgetDecision(max(0.0, cash - reserve),
                          0.0 if cash < reserve else min(500.0, mkt),
                          "normal", "deterministic reserve rule")


def _coordinator_llm(obs: dict, fc_today: float,
                     requested_mkt: float) -> BudgetDecision:
    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        return _fallback_budget(obs, requested_mkt)
    payload = {"cash": obs.get("cash"),
               "days_remaining": obs.get("days_remaining"),
               "yesterday_revenue": obs.get("yesterday_revenue"),
               "yesterday_total_costs": obs.get("yesterday_total_costs"),
               "reputation_band": obs.get("reputation_band"),
               "model_forecast_covers": round(fc_today),
               "requested_marketing": requested_mkt}
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
        return _fallback_budget(obs, requested_mkt)


def _reallocate(mix: dict, robust: set) -> dict:
    rob = [d for d in mix if d in robust]
    if not rob or len(rob) == len(mix):
        return dict(mix)
    fragile_mass = sum(s for d, s in mix.items() if d not in robust)
    if fragile_mass <= 1e-9:
        return dict(mix)
    rbase = sum(mix[d] for d in rob)
    out = {}
    for d, s in mix.items():
        if d in robust:
            add = (fragile_mass * (mix[d] / rbase) if rbase > 1e-9
                   else fragile_mass / len(rob))
            out[d] = s + add
        else:
            out[d] = s
    return out


def strategy(observation: dict, day: int) -> list[dict]:
    actions: list[dict] = []

    mem = Memory.from_notes(observation.get("notes"))
    mem.update(observation, day)
    regime = mem.regime()
    shelf = mem.shelf or None        # learned intrinsic shelf life

    wxf = observation.get("weather_forecast") or []
    wx_today = observation.get("weather_today")
    fc_today = mem.forecast((day - 1) % 7, wx_today)
    fc_tom = mem.forecast(day % 7, wxf[0] if len(wxf) > 0 else wx_today)
    fc_d2 = mem.forecast((day + 1) % 7, wxf[1] if len(wxf) > 1 else wx_today)
    horizon_peak = max(fc_today, fc_tom, fc_d2)
    supply_covers = horizon_peak * SAFETY_BAND
    ops_covers = max(fc_today, fc_tom)

    active = observation.get("active_menu", [])
    mix = mem.forecast_mix(active)

    horizon_wds = [(day - 1) % 7, day % 7, (day + 1) % 7]
    worst_wd, worst_robust, worst_n = horizon_wds[0], set(), 99
    for wd in horizon_wds:
        rob, _frag = supply_agent.dish_sourceability(observation, wd, shelf)
        rob_active = rob & set(active)
        if len(rob_active) < worst_n:
            worst_wd, worst_robust, worst_n = wd, rob_active, len(rob_active)
    realloc = _reallocate(mix, worst_robust)
    supply_by_dish = {d: max(mix.get(d, 0.0), realloc.get(d, 0.0))
                      * supply_covers for d in mix}

    today_robust, today_fragile = supply_agent.dish_sourceability(
        observation, (day - 1) % 7, shelf)

    model_view = {"covers_today": round(fc_today),
                  "horizon_peak": round(horizon_peak),
                  "regime": regime,
                  "robust_dishes_today": sorted(today_robust & set(active)),
                  "fragile_dishes_today": sorted(today_fragile & set(active)),
                  "dish_share": {d: round(s, 2) for d, s in mix.items()}}
    demand = demand_agent.propose(observation, day, mem.summary(), model_view)
    actions += demand.actions
    mem.record_forecast(fc_today)

    budget = _coordinator_llm(observation, fc_today, demand.requested_marketing)
    cash = float(observation.get("cash", 0.0))
    est_need = _rough_supply_estimate(observation, supply_by_dish)
    if regime in ("ramp", "surge", "recover"):
        est_need *= SURGE_BUDGET_MULT
    floor = min(max(0.0, cash - HARD_RESERVE), est_need)
    supply_budget = max(budget.supply_budget, floor)

    supply = supply_agent.propose(observation, day, supply_by_dish,
                                  supply_budget, regime, shelf)
    actions += supply.actions

    ops = operations_agent.propose(observation, day, ops_covers,
                                   budget.staff_bias)
    actions += ops.actions

    if budget.marketing_spend > 0:
        actions.append({"tool": "set_marketing_spend",
                        "args": {"amount": round(budget.marketing_spend, 2)}})

    actions.append({"tool": "save_notes", "args": {"text": mem.to_notes()}})

    if DEBUG:
        wn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        print(f"\n===== day {day} ({wn[(day-1)%7]}) "
              f"cash={observation.get('cash')} regime={regime} "
              f"acc={mem.bias:.2f} =====")
        print(f"  FORECAST today={fc_today:.0f} hpeak={horizon_peak:.0f} "
              f"supply_covers={supply_covers:.0f} ops={ops_covers:.0f}")
        print(f"  ROBUST today={sorted(today_robust & set(active))}")
        print(f"  FRAGILE today={sorted(today_fragile & set(active))}")
        print(f"  horizon constrained={wn[worst_wd]} "
              f"robust={sorted(worst_robust)}")
        print(f"  SUPPLY spend={supply.spend} budget={supply_budget:.0f} "
              f"orders={len(supply.actions)} at_risk={supply.at_risk}")
        print(f"  OPS staff={ops.staff_level} :: {ops.reasoning}")
        print(f"  shelf_known={sorted(mem.shelf)} notes={len(mem.to_notes())}c")
    return actions


if __name__ == "__main__":
    result = run_game(strategy, team_name="freaky-flamingos", seed=7)