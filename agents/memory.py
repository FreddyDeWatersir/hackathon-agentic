"""memory.py — bounded cross-day memory.

The agents are individually memoryless and reactive, so the system dies
the same way every run: it cannot see demand ramping (92→98→118) and
cannot pre-position before the surge. This holds the *little* cross-day
state that changes decisions.

HARD CONSTRAINT: fixed dimension. Rolling windows + EWMA scalars, never
an append-only log. Serialized it is ~600 chars (save_notes allows 4000).

OWNERSHIP: the coordinator rebuilds this from observation["notes"] every
turn, updates it, passes it READ-ONLY to the agents, and re-serializes
it. It is never held as in-process state — notes is per-game by the
contract, so this is immune to the parallel-evaluate cross-game hazard
that made us keep the agents stateless.

SELF-PROTECTION: a 0-cover day caused by an empty kitchen is NOT a demand
observation. Those days are forward-filled, not recorded, so regime/bias
are not poisoned by stockout zeros.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

WINDOW = 7          # rolling history length (fixed)
EWMA_A = 0.5        # weight on the newest sample
NO_SERVICE_ABS = 5  # covers ≤ this + Many walkouts ⇒ a stockout, not demand


@dataclass
class Memory:
    day: int = 0
    covers: list = field(default_factory=list)          # valid actual covers
    pending_forecast: float = 0.0                       # awaiting its actual
    bias: float = 1.0                                   # EWMA actual/forecast
    dish: dict = field(default_factory=dict)            # dish -> EWMA units
    supplier_fill: dict = field(default_factory=dict)   # supplier -> EWMA fill
    peak: float = 0.0
    starved_days: int = 0

    # ---------- lifecycle ----------
    @classmethod
    def from_notes(cls, notes: str | None) -> "Memory":
        try:
            d = json.loads(notes) if notes else {}
            if d.get("_m") != 1:
                raise ValueError
            return cls(day=d["day"], covers=list(d["c"]),
                       pending_forecast=d["pf"], bias=d["b"],
                       dish=dict(d["d"]), supplier_fill=dict(d["s"]),
                       peak=d["pk"], starved_days=d["sd"])
        except Exception:
            return cls()

    def to_notes(self) -> str:
        return json.dumps({
            "_m": 1, "day": self.day,
            "c": [round(x, 1) for x in self.covers[-WINDOW:]],
            "pf": round(self.pending_forecast, 1),
            "b": round(self.bias, 3),
            "d": {k: round(v, 1) for k, v in list(self.dish.items())[:12]},
            "s": {k: round(v, 2) for k, v in list(self.supplier_fill.items())[:8]},
            "pk": round(self.peak, 1), "sd": self.starved_days,
        })[:3500]

    # ---------- update from yesterday's results ----------
    def update(self, obs: dict, day: int) -> None:
        self.day = day
        ss = obs.get("service_summary", {}) or {}
        covers_y = float(ss.get("total_covers") or 0.0)
        walkout = str(ss.get("walkout_band", "None"))

        # Supplier fill rate (bounded EWMA) — learn who is flaky.
        for dh in obs.get("delivery_history", []) or []:
            sup = dh.get("supplier")
            ordered = float(dh.get("ordered_kg") or 0.0)
            got = float(dh.get("delivered_kg") or 0.0)
            if sup and ordered > 0:
                r = max(0.0, min(1.0, got / ordered))
                prev = self.supplier_fill.get(sup, r)
                self.supplier_fill[sup] = EWMA_A * r + (1 - EWMA_A) * prev

        no_service = (day > 1 and covers_y <= NO_SERVICE_ABS
                      and walkout in ("Some", "Many"))
        if no_service:
            self.starved_days += 1
            return  # do NOT record stockout zeros as demand
        self.starved_days = 0

        if day > 1:
            self.covers.append(covers_y)
            self.covers = self.covers[-WINDOW:]
            self.peak = max(self.peak, covers_y)
            if self.pending_forecast > 1e-6:
                r = covers_y / self.pending_forecast
                self.bias = EWMA_A * r + (1 - EWMA_A) * self.bias
            for dn, n in (ss.get("dishes_sold", {}) or {}).items():
                prev = self.dish.get(dn, float(n))
                self.dish[dn] = EWMA_A * float(n) + (1 - EWMA_A) * prev

    def record_forecast(self, f: float) -> None:
        self.pending_forecast = float(f)

    # ---------- derived signals ----------
    def regime(self) -> str:
        if self.starved_days >= 1:
            return "recover"          # just came off a stockout — refill hard
        c = self.covers
        if len(c) < 2:
            return "stable"
        a, b = c[-2], c[-1]
        if b >= a * 1.12:
            return "surge"
        if b >= a * 1.04:
            return "ramp"
        if b <= a * 0.85:
            return "drop"
        return "stable"

    def corrected(self, raw: float) -> float:
        """Bias-correct a forecast, clamped so a noisy ratio can't explode it."""
        return float(raw) * max(0.7, min(2.5, self.bias))

    def summary(self) -> dict:
        """Compact, human-readable block for the demand LLM prompt."""
        return {
            "recent_actual_covers": [round(x) for x in self.covers[-5:]],
            "your_forecast_bias": round(self.bias, 2),
            "bias_hint": ("you tend to UNDER-forecast — scale up"
                          if self.bias > 1.15 else
                          "you tend to OVER-forecast — scale down"
                          if self.bias < 0.85 else "forecast is well-calibrated"),
            "regime": self.regime(),
            "peak_covers_seen": round(self.peak),
        }