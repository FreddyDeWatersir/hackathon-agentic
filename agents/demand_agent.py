"""demand_agent.py — LLM-driven demand specialist (memory-aware).

Unchanged from before except: it now receives a compact MEMORY block
(recent actual covers, its own forecast bias, the regime) so it can
ground forecasts in the real trajectory instead of guessing from a
single snapshot. This is the proximate fix for the chronic
under-forecast that starves supply before the surge.

Fallback unchanged: no key / bad JSON ⇒ persistence forecast, no
menu/price changes, never crashes the game.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import litellm

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")

SYSTEM_PROMPT = """\
You are the demand manager of a 22-table Italian restaurant in a 30-day
survival game. You set prices, the active menu, happy hour, and the
daily special — and you forecast tomorrow's demand.

A MEMORY block gives you recent ACTUAL covers, your own forecast bias,
and the current regime. Your forecast MUST be consistent with that
trajectory: if recent actuals were ~100 and rising, do NOT forecast 18.
If your bias says you under-forecast, scale up accordingly.

Hard rules:
- Prices strictly INSIDE 0.8x-1.2x base (avoid the exact boundary).
- Active menu ≥ 5 dishes (exact names).
- Reputation is sticky and walkouts compound; don't chase volume you
  cannot serve. A dish whose ingredient is about to expire is a good
  daily_special pick.

Respond with ONLY this JSON, no prose, no markdown:
{
  "actions": [
    {"tool": "set_price", "args": {"dish": "...", "price": 0.0}},
    {"tool": "run_happy_hour", "args": {}},
    {"tool": "offer_daily_special", "args": {"dish": "..."}},
    {"tool": "set_menu", "args": {"dishes": ["...", "..."]}}
  ],
  "requested_marketing": 0,
  "forecast": {
    "expected_covers": 0,
    "by_dish": {"Dish Name": 0},
    "confidence": "low|med|high",
    "reasoning": "one short sentence"
  }
}
by_dish must cover every dish you expect to sell and sum ≈ expected_covers."""


@dataclass
class DemandResult:
    actions: list[dict] = field(default_factory=list)
    expected_covers: float = 0.0
    by_dish: dict[str, float] = field(default_factory=dict)
    confidence: str = "low"
    requested_marketing: float = 0.0
    reasoning: str = ""


def _persistence_forecast(obs: dict) -> tuple[float, dict[str, float]]:
    ss = obs.get("service_summary", {}) or {}
    sold = {k: float(v) for k, v in (ss.get("dishes_sold", {}) or {}).items()}
    if sold:
        return float(ss.get("total_covers", sum(sold.values()))), sold
    active = obs.get("active_menu", []) or []
    if not active:
        return 0.0, {}
    each = 90.0 / len(active)
    return 90.0, {d: each for d in active}


def _compact_observation(obs: dict) -> dict:
    inv_expiry = {
        i["ingredient"]: min((b.get("expires_in_days", 99)
                              for b in i.get("batches", []) or []), default=99)
        for i in obs.get("inventory", []) or []
    }
    ss = obs.get("service_summary", {}) or {}
    return {
        "day": obs.get("day"),
        "day_of_week": obs.get("day_of_week"),
        "days_remaining": obs.get("days_remaining"),
        "yesterday_revenue": obs.get("yesterday_revenue"),
        "yesterday_covers": ss.get("total_covers"),
        "yesterday_walkout_band": ss.get("walkout_band"),
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
            model=MODEL, temperature=0.3, max_tokens=1200,
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


def propose(obs: dict, day: int, mem_summary: dict | None = None) -> DemandResult:
    payload = {"MEMORY": mem_summary or {},
               "OBSERVATION": _compact_observation(obs)}
    user = f"Day {day}/30.\n{json.dumps(payload, indent=2)}"
    data = _call_llm(SYSTEM_PROMPT, user)

    fb_covers, fb_mix = _persistence_forecast(obs)
    if not isinstance(data, dict):
        return DemandResult(by_dish=fb_mix, expected_covers=fb_covers,
                            confidence="low", reasoning="LLM unavailable")

    fc = data.get("forecast", {}) if isinstance(data.get("forecast"), dict) else {}
    by_dish = fc.get("by_dish")
    if not isinstance(by_dish, dict) or not by_dish:
        by_dish, ec = fb_mix, fb_covers
    else:
        by_dish = {k: float(v) for k, v in by_dish.items()}
        ec = float(fc.get("expected_covers") or sum(by_dish.values()))

    actions = data.get("actions")
    if not isinstance(actions, list):
        actions = []

    try:
        req_mkt = float(data.get("requested_marketing", 0.0) or 0.0)
    except (TypeError, ValueError):
        req_mkt = 0.0

    return DemandResult(
        actions=actions, expected_covers=ec, by_dish=by_dish,
        confidence=str(fc.get("confidence", "low")),
        requested_marketing=max(0.0, min(500.0, req_mkt)),
        reasoning=str(fc.get("reasoning", "")),
    )