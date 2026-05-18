"""memory.py — two-tier memory for the multi-agent restaurant system.

Tier 1 — In-process (fast, full detail):
    _GAME_HISTORY: a module-level list that grows by one entry each turn.
    Lives as long as the Python process; ideal for a single 30-day game run.

Tier 2 — Persisted (survives restarts):
    A compressed JSON blob written via save_notes at the end of every turn
    and read back from observation["notes"] at the start of the next turn.
    Holds the last 7 days of detail plus derived weekly averages.

All public helpers read from _GAME_HISTORY (in-process) by default but
accept an explicit history list for testability.
"""
from __future__ import annotations

import json
from collections import defaultdict

# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------
_GAME_HISTORY: list[dict] = []

_DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
_DOW_IDX = {d: i for i, d in enumerate(_DAYS)}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def record_turn(
    obs: dict,
    day: int,
    forecast_covers: float,
    demand_conf: str,
    budget_supply: float,
    budget_mkt: float,
    staff_bias: str,
    staff_level: int,
    supply_spend: float,
    supply_at_risk: list[str],
    happy_hour: bool,
) -> None:
    """Append a compact summary of this turn to the in-process history.

    Call this AFTER all agents have made their decisions so the 'planned'
    fields are final.  The 'actuals' fields come from the observation, which
    always reflects yesterday's service results.
    """
    ss = obs.get("service_summary", {}) or {}
    cb = obs.get("cost_breakdown", {}) or {}

    today_idx = _DOW_IDX.get(obs.get("day_of_week", ""), -1)
    yesterday_dow = _DAYS[(today_idx - 1) % 7] if today_idx >= 0 else None

    _GAME_HISTORY.append({
        "day": day,
        "today_dow": obs.get("day_of_week"),
        "cash": float(obs.get("cash") or 0),
        "rep": obs.get("reputation_band"),
        # Yesterday's actual results (present in every observation except day 1)
        "covers": float(ss.get("total_covers") or 0),
        "covers_dow": yesterday_dow,   # DOW those covers happened on
        "revenue": float(obs.get("yesterday_revenue") or 0),
        "costs": float(obs.get("yesterday_total_costs") or 0),
        "walkout": ss.get("walkout_band"),
        "mkt_actual": float(cb.get("marketing") or 0),
        # Today's plan
        "forecast": round(forecast_covers, 1),
        "conf": demand_conf,
        "staff": staff_level,
        "bias": staff_bias,
        "supply_spend": round(supply_spend, 2),
        "mkt_plan": round(budget_mkt, 2),
        "happy_hour": happy_hour,
        "at_risk": supply_at_risk[:5],  # cap to keep notes small
        # Whether yesterday's service ran out of any dish (supply-driven walkout
        # signal). Stored so we can distinguish staffing vs supply failures
        # when reviewing historical walkout_band entries.
        "dishes_unavailable": bool(ss.get("dishes_unavailable_at")),
        "substitutions": int(ss.get("substitution_count", 0)),
    })


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_history() -> list[dict]:
    """Return the full in-process history (read-only reference)."""
    return _GAME_HISTORY


def get_recent(n: int = 7, history: list[dict] | None = None) -> list[dict]:
    """Last n entries from history."""
    h = history if history is not None else _GAME_HISTORY
    return h[-n:] if len(h) >= n else h[:]


def dow_cover_averages(history: list[dict] | None = None) -> dict[str, float]:
    """Average actual covers per day-of-week (open days only).

    Uses 'covers_dow' so averages are correctly attributed even though the
    observation arrives one day later.  Closed days (always 0) are excluded
    — use known_closed_dows() to detect those separately.
    """
    h = history if history is not None else _GAME_HISTORY
    buckets: dict[str, list[float]] = defaultdict(list)
    for entry in h:
        dow = entry.get("covers_dow")
        covers = entry.get("covers", 0)
        if dow and covers > 0:
            buckets[dow].append(covers)
    return {d: round(sum(v) / len(v), 1) for d, v in buckets.items()}


def known_closed_dows(min_observations: int = 1,
                      history: list[dict] | None = None) -> set[str]:
    """DOWs that have ONLY ever had 0 covers in our history.

    A DOW qualifies as 'closed' once we have seen it at least
    `min_observations` times and every single sighting was 0 covers.
    One positive-cover observation immediately removes it from the set.
    Effective from day 8 onward for Sundays (first sighting on Monday obs).
    """
    h = history if history is not None else _GAME_HISTORY
    dow_zero_count: dict[str, int] = {}
    dow_positive: set[str] = set()
    for entry in h:
        dow = entry.get("covers_dow")
        if not dow:
            continue
        if entry.get("covers", 0) > 0:
            dow_positive.add(dow)
        else:
            dow_zero_count[dow] = dow_zero_count.get(dow, 0) + 1
    return {
        d for d, cnt in dow_zero_count.items()
        if cnt >= min_observations and d not in dow_positive
    }


def upcoming_high_demand_dows(lookahead: int = 3,
                               high_threshold: float = 140.0,
                               history: list[dict] | None = None) -> list[str]:
    """DOWs occurring within the next `lookahead` days whose historical avg
    covers exceed `high_threshold`, ordered by proximity (soonest first).

    Used by the critic to pre-position staff before a predictable spike day.
    Returns an empty list when history is too short to have reliable averages.
    """
    avgs = dow_cover_averages(history)
    h = history if history is not None else _GAME_HISTORY
    if not h:
        return []
    today_dow = h[-1].get("today_dow", "")
    today_idx = _DOW_IDX.get(today_dow, -1)
    if today_idx < 0:
        return []
    result = []
    for offset in range(1, lookahead + 1):
        future_dow = _DAYS[(today_idx + offset) % 7]
        if avgs.get(future_dow, 0) >= high_threshold:
            result.append(future_dow)
    return result


def happy_hour_streak(history: list[dict] | None = None) -> int:
    """Number of consecutive turns (ending with the most recent) where
    happy_hour was planned True."""
    h = history if history is not None else _GAME_HISTORY
    streak = 0
    for entry in reversed(h):
        if entry.get("happy_hour"):
            streak += 1
        else:
            break
    return streak


def consecutive_zero_open_covers(history: list[dict] | None = None) -> int:
    """Count of consecutive trailing open days (non-closed DOWs) with 0 covers.

    Returns 0 as soon as a day with covers > 0 is reached (iterating backwards).
    Used by the critic to distinguish a stock-out spiral from genuine
    understaffing — if this is >= 2 and walkout=Many, it's a supply failure,
    not a staffing failure, so escalating staff only burns cash faster.
    """
    h = history if history is not None else _GAME_HISTORY
    closed = known_closed_dows(history=h)
    count = 0
    for entry in reversed(h):
        dow = entry.get("covers_dow")
        if dow in closed:
            continue  # skip closed days (Sunday etc.)
        if entry.get("covers", 0) == 0:
            count += 1
        else:
            break
    return count


def dow_walkout_rates(history: list[dict] | None = None) -> dict[str, float]:
    """Fraction of open days per DOW where walkout was 'Many' or 'Some'.

    Only includes DOWs with at least 2 observations so early noise doesn't
    skew results. Closed DOWs are excluded. Used by the critic to pre-empt
    chronically bad days rather than reacting after damage is done.
    """
    h = history if history is not None else _GAME_HISTORY
    closed = known_closed_dows(history=h)
    total: dict[str, int] = {}
    bad: dict[str, int] = {}
    for entry in h:
        dow = entry.get("covers_dow")
        if not dow or dow in closed:
            continue
        total[dow] = total.get(dow, 0) + 1
        if entry.get("walkout") in ("Many", "Some"):
            bad[dow] = bad.get(dow, 0) + 1
    return {
        d: round(bad.get(d, 0) / cnt, 2)
        for d, cnt in total.items()
        if cnt >= 2
    }


def chronic_at_risk_ingredients(
    min_occurrences: int = 2,
    window: int = 7,
    history: list[dict] | None = None,
) -> list[str]:
    """Ingredients in 'at_risk' at least min_occurrences times in the last
    `window` turns, sorted by frequency (most chronic first).

    A repeated at-risk signal means the supply agent keeps under-ordering
    the same ingredient. The critic should flag these so the coordinator
    allocates dedicated budget rather than treating each shortage as one-off.
    """
    h = history if history is not None else _GAME_HISTORY
    recent = h[-window:] if len(h) >= window else h[:]
    counts: dict[str, int] = {}
    for entry in recent:
        for ing in entry.get("at_risk", []):
            counts[ing] = counts.get(ing, 0) + 1
    return sorted(
        (ing for ing, cnt in counts.items() if cnt >= min_occurrences),
        key=lambda i: -counts[i],
    )


def forecast_accuracy_recent(
    n: int = 5, history: list[dict] | None = None
) -> float:
    """Average ratio of actual covers to forecast covers over the last n open turns.

    > 1.0  demand agent under-forecasts (ops/supply are under-sized).
    < 1.0  demand agent over-forecasts (over-staffed, excess supply, waste).
    Returns 1.0 when there is no usable data. Closed DOWs and zero forecasts
    are excluded.
    """
    h = history if history is not None else _GAME_HISTORY
    closed = known_closed_dows(history=h)
    ratios: list[float] = []
    for entry in reversed(h):
        if len(ratios) >= n:
            break
        if entry.get("covers_dow") in closed:
            continue
        fc = entry.get("forecast", 0)
        cov = entry.get("covers", 0)
        if fc > 0:
            ratios.append(cov / fc)
    if not ratios:
        return 1.0
    return round(sum(ratios) / len(ratios), 2)


def reputation_trend(history: list[dict] | None = None) -> str:
    """Direction of reputation over the last 3 recorded turns.

    Returns 'improving', 'stable', or 'declining'.
    'declining' is an emergency: tighten staffing and suppress walkout risk.
    """
    _BANDS = {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Excellent": 4}
    h = history if history is not None else _GAME_HISTORY
    scored = [_BANDS[e["rep"]] for e in h[-3:] if e.get("rep") in _BANDS]
    if len(scored) < 2:
        return "stable"
    if scored[-1] > scored[0]:
        return "improving"
    if scored[-1] < scored[0]:
        return "declining"
    return "stable"


def recent_profit_trend(n: int = 5, history: list[dict] | None = None) -> float:
    """Average daily P&L over the last n turns (revenue - costs)."""
    h = get_recent(n, history)
    if not h:
        return 0.0
    profits = [e.get("revenue", 0) - e.get("costs", 0) for e in h]
    return round(sum(profits) / len(profits), 2)


# ---------------------------------------------------------------------------
# Persisted notes blob
# ---------------------------------------------------------------------------

def build_notes(
    day: int,
    obs: dict,
    forecast_covers: float,
    demand_conf: str,
    budget_supply: float,
    budget_mkt: float,
    staff_bias: str,
    staff_level: int,
    supply_spend: float,
    supply_at_risk: list[str],
    happy_hour: bool,
    critique_flags: list[str] | None = None,
) -> str:
    """Serialize the memory state to a ≤4000-char JSON string for save_notes."""
    compact_recent = [
        {
            "d": e["day"],
            "dow": e.get("covers_dow"),
            "cov": e["covers"],
            "rev": e["revenue"],
            "pnl": round(e["revenue"] - e["costs"], 0),
            "wo": e["walkout"],
            "hh": e["happy_hour"],
        }
        for e in get_recent(7)
    ]

    blob = {
        "day": day,
        "cash": obs.get("cash"),
        "rep": obs.get("reputation_band"),
        "dow_avg": dow_cover_averages(),
        "closed_dows": list(known_closed_dows()),
        "hh_streak": happy_hour_streak(),
        "avg_pnl_5d": recent_profit_trend(5),
        "recent": compact_recent,
        "today": {
            "forecast": round(forecast_covers, 1),
            "conf": demand_conf,
            "staff": staff_level,
            "bias": staff_bias,
            "supply_spend": supply_spend,
            "mkt": round(budget_mkt, 2),
            "hh": happy_hour,
            "at_risk": supply_at_risk[:5],
        },
        "critic_flags": (critique_flags or [])[:5],
        # Derived analytics — gives the LLM critic and coordinator a richer
        # picture without requiring them to re-derive from raw history.
        "dow_walkout_rates": dow_walkout_rates(),
        "chronic_at_risk": chronic_at_risk_ingredients(min_occurrences=2, window=7),
        "forecast_accuracy_5d": forecast_accuracy_recent(5),
        "rep_trend": reputation_trend(),
    }
    return json.dumps(blob)[:4000]


def load_notes(notes_str: str | None) -> dict:
    """Parse the persisted notes blob.  Returns empty dict on any failure."""
    if not notes_str:
        return {}
    try:
        return json.loads(notes_str)
    except Exception:
        return {}
