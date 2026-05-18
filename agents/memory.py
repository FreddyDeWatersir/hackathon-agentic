"""memory.py — bounded cross-day memory + deterministic demand model.

CHANGE (regression fix): now also learns each ingredient's INTRINSIC
shelf life. A depleted inventory entry reports shelf_life_days = 0, which
made the #2 sourceability oracle (which read shelf life from current
inventory) collapse — every dish marked fragile once stock ran low.
Shelf life is a static property, so we record the max sane value ever
seen per ingredient and the oracle reads THAT, immune to the zeros.

Otherwise unchanged: rebuilt from observation["notes"] every turn
(parallel-safe), serialized back, bounded (~850 chars). The demand model
(weekday baseline × weather × trend) is the single source of truth fed
to supply/ops.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

WINDOW = 7
A = 0.5
A_BUCKET = 0.4
P0 = 90.0
NO_SERVICE_ABS = 5
WF_MIN, WF_MAX = 0.6, 1.5
TR_MIN, TR_MAX = 0.6, 1.8
FC_MIN = 10.0


@dataclass
class Memory:
    day: int = 0
    covers: list = field(default_factory=list)
    pending_forecast: float = 0.0
    bias: float = 1.0
    dish: dict = field(default_factory=dict)
    supplier_fill: dict = field(default_factory=dict)
    peak: float = 0.0
    starved_days: int = 0
    wd_base: dict = field(default_factory=dict)
    wx: dict = field(default_factory=dict)
    trend: float = 1.0
    lvl: float = 0.0
    pw: str = ""
    shelf: dict = field(default_factory=dict)        # intrinsic shelf life

    # ---------- lifecycle ----------
    @classmethod
    def from_notes(cls, notes: str | None) -> "Memory":
        try:
            d = json.loads(notes) if notes else {}
            if d.get("_m") != 3:
                raise ValueError
            return cls(day=d["day"], covers=list(d["c"]),
                       pending_forecast=d["pf"], bias=d["b"],
                       dish=dict(d["d"]), supplier_fill=dict(d["s"]),
                       peak=d["pk"], starved_days=d["sd"],
                       wd_base=dict(d["wb"]), wx=dict(d["wx"]),
                       trend=d["tr"], lvl=d["lv"], pw=d["pw"],
                       shelf=dict(d["sh"]))
        except Exception:
            return cls()

    def to_notes(self) -> str:
        return json.dumps({
            "_m": 3, "day": self.day,
            "c": [round(x, 1) for x in self.covers[-WINDOW:]],
            "pf": round(self.pending_forecast, 1), "b": round(self.bias, 3),
            "d": {k: round(v, 1) for k, v in list(self.dish.items())[:12]},
            "s": {k: round(v, 2) for k, v in list(self.supplier_fill.items())[:8]},
            "pk": round(self.peak, 1), "sd": self.starved_days,
            "wb": {k: round(v, 1) for k, v in list(self.wd_base.items())[:7]},
            "wx": {k: round(v, 3) for k, v in list(self.wx.items())[:5]},
            "tr": round(self.trend, 3), "lv": round(self.lvl, 1),
            "pw": self.pw,
            "sh": {k: round(v, 1) for k, v in list(self.shelf.items())[:14]},
        })[:3500]

    # ---------- demand model ----------
    def _fc_max(self) -> float:
        return max(300.0, 1.15 * self.peak)

    def forecast(self, weekday: int, weather: str | None) -> float:
        base = self.wd_base.get(str(weekday)) or self.lvl or P0
        wf = max(WF_MIN, min(WF_MAX, self.wx.get(weather or "", 1.0)))
        tr = max(TR_MIN, min(TR_MAX, self.trend))
        return float(max(FC_MIN, min(self._fc_max(), base * wf * tr)))

    def forecast_mix(self, active_menu: list[str]) -> dict[str, float]:
        active = list(active_menu or [])
        if not active:
            return {}
        shares = {d: self.dish.get(d, 0.0) for d in active}
        tot = sum(shares.values())
        if tot <= 1e-6:
            return {d: 1.0 / len(active) for d in active}
        avg = tot / len(active)
        for d in active:
            if shares[d] <= 1e-6:
                shares[d] = avg
        tot = sum(shares.values())
        return {d: shares[d] / tot for d in active}

    # ---------- update ----------
    def update(self, obs: dict, day: int) -> None:
        self.day = day
        ss = obs.get("service_summary", {}) or {}
        covers_y = float(ss.get("total_covers") or 0.0)
        walkout = str(ss.get("walkout_band", "None"))
        today_weather = str(obs.get("weather_today", "") or "")

        # Intrinsic shelf life: max sane value ever seen (ignore the 0s a
        # depleted entry reports). Runs every turn, even no-service days.
        for i in obs.get("inventory", []) or []:
            ing = i.get("ingredient")
            sd = i.get("shelf_life_days")
            try:
                sd = float(sd)
            except (TypeError, ValueError):
                continue
            if ing and sd > 0:
                self.shelf[ing] = max(self.shelf.get(ing, 0.0), sd)

        for dh in obs.get("delivery_history", []) or []:
            sup = dh.get("supplier")
            ordered = float(dh.get("ordered_kg") or 0.0)
            got = float(dh.get("delivered_kg") or 0.0)
            if sup and ordered > 0:
                r = max(0.0, min(1.0, got / ordered))
                prev = self.supplier_fill.get(sup, r)
                self.supplier_fill[sup] = A * r + (1 - A) * prev

        no_service = (day > 1 and covers_y <= NO_SERVICE_ABS
                      and walkout in ("Some", "Many"))
        if no_service:
            self.starved_days += 1
            self.pw = today_weather
            return
        self.starved_days = 0

        if day > 1:
            wd_y = (day - 2) % 7
            pred_full = self.forecast(wd_y, self.pw)
            self.covers.append(covers_y)
            self.covers = self.covers[-WINDOW:]
            self.peak = max(self.peak, covers_y)
            self.lvl = A * covers_y + (1 - A) * (self.lvl or covers_y)
            r = covers_y / max(pred_full, 1.0)
            self.trend = max(TR_MIN, min(TR_MAX, A * r + (1 - A) * self.trend))
            k = str(wd_y)
            self.wd_base[k] = (A_BUCKET * covers_y
                               + (1 - A_BUCKET) * self.wd_base.get(k, covers_y))
            wbase = self.wd_base.get(k) or self.lvl or P0
            if self.pw:
                wf_obs = covers_y / max(wbase, 1.0)
                prev = self.wx.get(self.pw, wf_obs)
                self.wx[self.pw] = max(WF_MIN, min(WF_MAX,
                                       A * wf_obs + (1 - A) * prev))
            if self.pending_forecast > 1e-6:
                self.bias = (A * (covers_y / self.pending_forecast)
                             + (1 - A) * self.bias)
            for dn, n in (ss.get("dishes_sold", {}) or {}).items():
                prev = self.dish.get(dn, float(n))
                self.dish[dn] = A * float(n) + (1 - A) * prev

        self.pw = today_weather

    def record_forecast(self, f: float) -> None:
        self.pending_forecast = float(f)

    def regime(self) -> str:
        if self.starved_days >= 1:
            return "recover"
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

    def summary(self) -> dict:
        return {
            "recent_actual_covers": [round(x) for x in self.covers[-5:]],
            "model_forecast_accuracy": round(self.bias, 2),
            "regime": self.regime(),
            "peak_covers_seen": round(self.peak),
        }