"""demand_agent.py — LLM-driven demand specialist.

Serves the coordinator. Owns: set_price, set_menu, run_happy_hour,
offer_daily_special. (Marketing spend is a budget item the coordinator
emits; this agent only *requests* an amount.)

Its second, equally important job: produce the demand FORECAST
(expected_covers + per-dish mix) that supply and operations both consume.
The forecast must reflect demand UNDER the actions it proposes.

If the LLM is unavailable or returns garbage, it falls back to a
persistence forecast (yesterday's dish mix) and makes no menu/price
changes — safe, never crashes the game.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import litellm

from agents import memory as mem

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")

SYSTEM_PROMPT = """\
You are the demand manager of a 22-table Italian restaurant in a 30-day
survival game. You set prices, the active menu, happy hour, and the daily
special to drive covers and revenue — and you forecast tomorrow's demand.

Hard rules:
- Prices must stay within 0.8x-1.2x each dish's base_price.
- The active menu must have at least 5 dishes (use exact names).
- Reputation is sticky and walkouts compound: do not chase volume you
  cannot serve. Modest, steady demand beats spiky demand.
- A dish whose ingredient is about to expire is a good daily_special pick.
- Use dow_cover_avg (historical covers by day-of-week) to calibrate your
  expected_covers forecast: if today is historically strong, forecast higher.
- If happy_hour_streak >= 3, do NOT include run_happy_hour — effectiveness
  has decayed and the critic will veto it anyway.

Respond with ONLY this JSON object, no prose, no markdown:
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
Only include actions you actually want to take. by_dish must cover every
dish you expect to sell and sum roughly to expected_covers."""


@dataclass
class DemandResult:
    actions: list[dict] = field(default_factory=list)
    expected_covers: float = 0.0
    by_dish: dict[str, float] = field(default_factory=dict)
    confidence: str = "low"
    requested_marketing: float = 0.0
    reasoning: str = ""


def _persistence_forecast(obs: dict) -> tuple[float, dict[str, float]]:
    """Fallback: assume tomorrow ≈ yesterday's realised dish mix.

    All active-menu dishes are always included with at least a small floor so
    the supply agent maintains stock for every dish — not just the ones that
    happened to sell yesterday (which collapses to a subset if some dishes ran
    out of ingredients mid-service and got 0 sales).
    """
    ss = obs.get("service_summary", {}) or {}
    sold = {k: float(v) for k, v in (ss.get("dishes_sold", {}) or {}).items()}
    active = obs.get("active_menu", []) or []
    if sold:
        total = sum(sold.values())
        # Floor: every active dish gets at least 5 % of average-per-dish covers
        # so supply never stops ordering for dishes that were temporarily zero.
        floor = max(1.0, total / max(len(active), 1) * 0.05)
        for dish in active:
            if dish not in sold:
                sold[dish] = floor
        return float(ss.get("total_covers", total)), sold
    # Day 1 / no history: spread a modest default over the active menu.
    if not active:
        return 0.0, {}
    each = 90.0 / len(active)
    return 90.0, {d: each for d in active}


def _compact_observation(obs: dict) -> dict:
    """Only what demand needs — keeps the prompt small and cheap."""
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
        "yesterday_dishes_sold": ss.get("dishes_sold"),
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
        # Long-term memory: day-of-week cover averages and promotion streak.
        "dow_cover_avg": mem.dow_cover_averages(),
        "happy_hour_streak": mem.happy_hour_streak(),
    }


def _call_llm(system: str, user: str) -> dict | None:
    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")):
        return None
        
    try:
        resp = litellm.completion(
            model=MODEL, temperature=0.3, max_tokens=1200,
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
        print(f"  [demand] LLM fallback: {e}")
        return None


def propose(obs: dict, day: int) -> DemandResult:
    user = (f"Day {day}/30. State:\n"
            f"{json.dumps(_compact_observation(obs), indent=2)}")
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
        actions=actions,
        expected_covers=ec,
        by_dish=by_dish,
        confidence=str(fc.get("confidence", "low")),
        requested_marketing=max(0.0, min(500.0, req_mkt)),
        reasoning=str(fc.get("reasoning", "")),
    )