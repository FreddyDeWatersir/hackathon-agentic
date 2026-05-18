"""critic_agent.py — post-assembly plan auditor.

Runs LAST in the pipeline, after all agents have proposed their actions.
Receives the combined plan and memory context, then checks for:

  1. CASH RUNWAY   — today's spends must leave enough cover for remaining days.
  2. REPUTATION RISK — lean staffing when yesterday saw high walkouts is forbidden,
                       EXCEPT when yesterday was a closed/0-cover day OR had any
                       stock-outs (walkouts in those cases are supply-driven, not
                       understaffing — escalating staff just burns cash).
  3. PROMOTION DECAY — happy_hour vetoed after 3+ consecutive days.
  4. DOW MISMATCH  — if forecast >> historical avg for this weekday, escalate bias,
                     UNLESS yesterday was 0-cover (the LLM systematically over-
                     forecasts the day after a closure with no recent open signal).
  5. SUPPLY AT RISK — surfaces warnings without overriding (budget may be exhausted).

Returns a CritiqueResult with optional overrides and a list of veto'd tools.
Falls back to deterministic rules if the LLM is unavailable.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import litellm

from agents import memory as mem

MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4.1-mini")

CRITIC_SYSTEM = """\
You are a risk auditor for a 30-day restaurant survival game.

You receive the PROPOSED plan (demand + supply + ops + coordinator) and
MEMORY (last 7 days of actuals, day-of-week cover averages, happy-hour streak,
known closed DOWs, and upcoming high-demand days within 2 days).
Your sole job: catch decisions that are individually reasonable but jointly
dangerous. Do NOT change decisions that look fine.

Audit checklist — only flag real issues:
1. CASH RUNWAY: (cash - supply_budget - marketing_spend) must cover at least
   min(days_remaining, 6) × estimated_daily_overhead. If not, cut supply_budget.
2. WALKOUT ESCALATION (HIGH PRIORITY — do not escalate falsely):
   - CRITICAL: if yesterday_covers == 0, IGNORE walkout_band entirely. Closed
     days and full stock-outs report "Many" walkouts as an artifact of customers
     finding the restaurant non-operational. NOT an understaffing signal.
   - CRITICAL: if yesterday_dishes_unavailable_at is non-empty, IGNORE walkout_band.
     The walkouts were caused by dishes running out, NOT understaffing.
     Escalating staff burns ~120 EUR per person; the real fix (supply) is
     outside your scope and adding staff to an empty pantry helps no one.
   - Otherwise: yesterday walkout_band = "Many" → override staff_bias to "safe".
   - Otherwise: yesterday walkout_band = "Some" AND staff_bias = "lean" → override to "normal".
3. PROMOTION DECAY: If happy_hour_streak >= 3 AND run_happy_hour is proposed,
   veto run_happy_hour.
4. DOW DEMAND LEVEL:
   - dow_cover_averages[today] >= 140 → staff_bias must be "safe".
   - dow_cover_averages[today] >= 100 → staff_bias must be at least "normal".
   - forecast_covers > 1.3 × dow_avg[today] → escalate one step (lean→normal or
     normal→safe). UNLESS yesterday_covers == 0 — after a closed day the LLM
     forecasts blindly off DOW priors and tends to overshoot; trust dow_avg
     instead of the surge.
5. PRE-POSITIONING: If upcoming_high_demand (within 2 days) is non-empty AND
   staff_bias is lean/normal → override to "safe" so staff can ramp up in time.
6. SUPPLY AT RISK: If at_risk is non-empty, add a warning flag (no budget override).

Respond with ONLY valid JSON, no prose:
{
  "approved": true,
  "override_supply_budget": null,
  "override_marketing_spend": null,
  "override_staff_bias": null,
  "veto_tools": [],
  "flags": [],
  "reasoning": "one short sentence"
}
null means no change. Only override when the checklist demands it."""


@dataclass
class CritiqueResult:
    approved: bool = True
    override_supply_budget: float | None = None
    override_marketing_spend: float | None = None
    override_staff_bias: str | None = None
    veto_tools: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Deterministic fallback (always runs if LLM is unavailable or fails)
# ---------------------------------------------------------------------------

def _fallback_critique(
    obs: dict,
    budget_supply: float,
    budget_mkt: float,
    staff_bias: str,
    forecast_covers: float,
    happy_hour: bool,
    proposed_tools: list[str],
) -> CritiqueResult:
    result = CritiqueResult()

    cash = float(obs.get("cash") or 0)
    staff = float(obs.get("staff_level") or 8)
    cost_per = float(obs.get("staff_cost_per_person") or 120.0)
    days_left = int(obs.get("days_remaining") or 1)
    ss = obs.get("service_summary", {}) or {}
    walkout = str(ss.get("walkout_band") or "None")
    yesterday_covers = float(ss.get("total_covers") or 0)
    dishes_unavail = ss.get("dishes_unavailable_at") or {}
    had_stockout = bool(dishes_unavail)
    today_dow = obs.get("day_of_week", "")
    dow_avg = mem.dow_cover_averages()
    hist_avg = dow_avg.get(today_dow, 0.0)

    # Track the effective bias as overrides accumulate.
    def effective() -> str:
        return result.override_staff_bias or staff_bias

    # 1. Cash runway.
    daily_overhead = staff * cost_per + 300.0
    runway_days = min(days_left, 6)
    min_reserve = daily_overhead * runway_days
    available_after = cash - budget_supply - budget_mkt
    # Only cut supply budget when cash is tight AND we are not already in a
    # stock-out spiral — cutting supply when the shelves are empty makes
    # things worse, not better.
    zero_open_streak = mem.consecutive_zero_open_covers()
    if available_after < min_reserve * 0.5 and zero_open_streak < 2:
        cut = max(0.0, cash - min_reserve - budget_mkt)
        result.override_supply_budget = cut
        result.flags.append(
            f"cash runway tight: available={available_after:.0f} "
            f"vs reserve={min_reserve:.0f}; cut supply_budget to {cut:.0f}"
        )

    # 2. Walkout escalation.
    #    "Many"  → must be safe regardless of current bias.
    #    "Some"  → must be at least normal, unless today is historically quiet.
    #
    # EXCEPTION 1: walkouts on a 0-cover day are a closed-day or total stock-out
    # artifact. Customers arrive, find no service, simulator records walkouts.
    # NOT understaffing — escalating burns ~360+ EUR for nothing.
    #
    # EXCEPTION 2 (NEW): if dishes_unavailable_at is non-empty, yesterday had a
    # partial stock-out. The walkouts were customers who couldn't get the dishes
    # they wanted, not customers who couldn't get a table. Escalating staff puts
    # extra bodies in an empty pantry — saw a 1,857 EUR cash drop on Day 20
    # of one run from exactly this pattern (Day 19 stock-out → Day 20 "safe"
    # staffing → 9 covers, massive waste).
    #
    # EXCEPTION 3: if we have seen 2+ consecutive open days with 0 covers, the
    # walkouts are caused by a stock-out spiral (no food to serve), NOT under-
    # staffing. Detect and flag.
    zero_cover_yesterday = yesterday_covers == 0

    if walkout == "Many" and effective() in ("lean", "normal"):
        if zero_cover_yesterday:
            result.flags.append(
                f"walkout=Many ignored: yesterday had 0 covers "
                f"(closed day or stock-out artifact, not understaffing)"
            )
        elif had_stockout:
            stockouts = list(dishes_unavail.keys())[:3]
            result.flags.append(
                f"walkout=Many ignored: yesterday had stock-outs on {stockouts} "
                f"— walkouts driven by missing dishes, not understaffing "
                f"(escalating staff onto an empty pantry burns cash)"
            )
        elif zero_open_streak >= 2:
            result.flags.append(
                f"walkout=Many but {zero_open_streak} consecutive zero-cover open days "
                f"detected — likely stock-out spiral, not understaffing; "
                f"skipping staff escalation to preserve cash for restocking"
            )
        else:
            result.override_staff_bias = "safe"
            result.flags.append(
                f"walkout=Many yesterday; escalated bias {staff_bias} → safe"
            )
    elif walkout == "Some" and effective() == "lean":
        if zero_cover_yesterday:
            result.flags.append(
                f"walkout=Some ignored: yesterday had 0 covers (closed/stock-out artifact)"
            )
        elif had_stockout:
            stockouts = list(dishes_unavail.keys())[:3]
            result.flags.append(
                f"walkout=Some ignored: yesterday had stock-outs on {stockouts} "
                f"— supply-driven, not staffing"
            )
        elif not (hist_avg > 0 and forecast_covers <= hist_avg * 0.6):
            result.override_staff_bias = "normal"
            result.flags.append(
                f"walkout=Some + lean bias; escalated to normal"
            )

    # 3. Promotion decay.
    streak = mem.happy_hour_streak()
    if happy_hour and streak >= 3:
        result.veto_tools.append("run_happy_hour")
        result.flags.append(
            f"run_happy_hour vetoed: streak={streak} consecutive days"
        )

    # 4. DOW demand level — use historical average to set the floor bias.
    #    >= 140 avg covers → safe  |  >= 100 avg → at least normal
    #    forecast >> hist_avg → escalate one step (lean→normal OR normal→safe)
    if hist_avg >= 140.0 and effective() in ("lean", "normal"):
        result.override_staff_bias = "safe"
        result.flags.append(
            f"high-demand DOW {today_dow} (hist={hist_avg:.0f}≥140); escalated to safe"
        )
    elif hist_avg >= 100.0 and effective() == "lean":
        result.override_staff_bias = "normal"
        result.flags.append(
            f"moderate-demand DOW {today_dow} (hist={hist_avg:.0f}≥100); escalated to normal"
        )
    elif hist_avg > 0 and forecast_covers > hist_avg * 1.3:
        # NEW: skip the surge rule when yesterday was 0-cover. The LLM's
        # forecast for Monday after Sunday is consistently ~90 when actuals
        # run ~50 — it has no fresh open-day signal and over-shoots. Trusting
        # the surge then over-staffs every single Monday.
        if zero_cover_yesterday:
            result.flags.append(
                f"forecast surge ({forecast_covers:.0f} vs hist {hist_avg:.0f}) "
                f"ignored: yesterday was 0-cover, LLM forecast unreliable"
            )
        elif effective() == "lean":
            result.override_staff_bias = "normal"
            result.flags.append(
                f"forecast ({forecast_covers:.0f}) > 130% of hist ({hist_avg:.0f}); lean→normal"
            )
        elif effective() == "normal":
            result.override_staff_bias = "safe"
            result.flags.append(
                f"forecast ({forecast_covers:.0f}) > 130% of hist ({hist_avg:.0f}); normal→safe"
            )

    # 5. Pre-positioning: high-demand day within 2 days → build staff now.
    #    Only fires if we still have enough history (≥7 days) to trust averages.
    upcoming = mem.upcoming_high_demand_dows(lookahead=2, high_threshold=140.0)
    if upcoming and len(mem.get_history()) >= 7 and effective() in ("lean", "normal"):
        result.override_staff_bias = "safe"
        result.flags.append(
            f"pre-positioning for {upcoming[0]} (high-demand, within 2 days); escalated to safe"
        )

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def audit(
    obs: dict,
    day: int,
    actions: list[dict],
    budget,
    forecast_covers: float,
    happy_hour: bool,
) -> CritiqueResult:
    """Audit the assembled plan and return a (possibly empty) CritiqueResult."""

    dow_avg = mem.dow_cover_averages()
    streak = mem.happy_hour_streak()
    recent = mem.get_recent(7)
    proposed_tools = [a.get("tool", "") for a in actions]

    if not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    ):
        return _fallback_critique(
            obs, budget.supply_budget, budget.marketing_spend,
            budget.staff_bias, forecast_covers, happy_hour, proposed_tools,
        )

    ss = obs.get("service_summary", {}) or {}
    staff = float(obs.get("staff_level") or 8)
    cost_per = float(obs.get("staff_cost_per_person") or 120.0)
    yesterday_covers = float(ss.get("total_covers") or 0)
    dishes_unavail = ss.get("dishes_unavailable_at") or {}
    payload = {
        "day": day,
        "today_dow": obs.get("day_of_week"),
        "cash": obs.get("cash"),
        "days_remaining": obs.get("days_remaining"),
        "estimated_daily_overhead": round(staff * cost_per + 300.0, 2),
        "yesterday_walkout_band": ss.get("walkout_band"),
        "yesterday_covers": yesterday_covers,
        # Surface the closed/stock-out artifact flags explicitly so the LLM
        # doesn't have to infer them. Both are independent reasons to ignore
        # the walkout_band signal.
        "yesterday_was_zero_cover_day": yesterday_covers == 0,
        "yesterday_dishes_unavailable_at": dishes_unavail,
        "yesterday_had_stockout": bool(dishes_unavail),
        "reputation_band": obs.get("reputation_band"),
        "proposed_supply_budget": round(budget.supply_budget, 2),
        "proposed_marketing_spend": round(budget.marketing_spend, 2),
        "proposed_staff_bias": budget.staff_bias,
        "proposed_tools": proposed_tools,
        "forecast_covers": round(forecast_covers, 1),
        "happy_hour_streak": streak,
        "dow_cover_averages": dow_avg,
        "known_closed_dows": list(mem.known_closed_dows()),
        "upcoming_high_demand_dows": mem.upcoming_high_demand_dows(
            lookahead=2, high_threshold=140.0
        ),
        "recent_7_days": [
            {k: v for k, v in e.items()
             if k in ("day", "covers_dow", "covers", "revenue", "costs",
                      "walkout", "happy_hour")}
            for e in recent
        ],
    }

    try:
        resp = litellm.completion(
            model=MODEL,
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": CRITIC_SYSTEM
                    + "\nOutput exactly valid JSON without trailing commas.",
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        txt = resp.choices[0].message.content.strip()
        if txt.startswith("```"):
            txt = txt.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if txt.startswith("json"):
                txt = txt[4:].strip()
        txt = re.sub(r',\s*\}', '}', txt)
        txt = re.sub(r',\s*\]', ']', txt)
        d = json.loads(txt)

        cash = float(obs.get("cash") or 0)
        ob_sup = d.get("override_supply_budget")
        ob_mkt = d.get("override_marketing_spend")
        ob_bias = d.get("override_staff_bias")

        return CritiqueResult(
            approved=bool(d.get("approved", True)),
            override_supply_budget=(
                max(0.0, min(cash, float(ob_sup))) if ob_sup is not None else None
            ),
            override_marketing_spend=(
                max(0.0, min(500.0, float(ob_mkt))) if ob_mkt is not None else None
            ),
            override_staff_bias=str(ob_bias) if ob_bias else None,
            veto_tools=list(d.get("veto_tools") or []),
            flags=list(d.get("flags") or []),
            reasoning=str(d.get("reasoning", "")),
        )

    except Exception as e:
        print(f"  [critic] LLM fallback: {e}")
        return _fallback_critique(
            obs, budget.supply_budget, budget.marketing_spend,
            budget.staff_bias, forecast_covers, happy_hour, proposed_tools,
        )