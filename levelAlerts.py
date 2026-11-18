"""Level-touch alerts: tell me when price REACHES a key level from levels.py.

Pure logic, no network and no config reads at call time beyond the defaults in
LevelAlertSettings. main.py feeds it the 5-minute frame it has ALREADY fetched
for the MACD analysis plus the SupportResistance it already built, so level
alerts cost zero extra Alpaca/Yahoo requests.

    events, new_state = check_levels(symbol, df_5m, sr, now, state, settings)
    text = format_events(symbol, events)

What counts as a touch
  A closed bar touches a level when the bar's price PATH comes within a
  tolerance (percent of the level) of it. The path is the bar's low..high
  extended to the previous bar's close, so a gap that jumps over a level
  between two bars still counts. Each event also says whether price came
  "from below" or "from above" the level.

Which bars are examined (state-driven, no re-alerts)
  * saved state:  bars that closed after the saved cursor (missed runs are
                  caught up, capped at catchup_min minutes back)
  * no state:     only bars that closed within max_age_min of `now`
  Per symbol/day the state remembers which level|session already alerted
  (optionally re-arming after rearm_min), and a daily alert cap.

Sessions (config.LEVEL_SESSIONS) decide which families are watched and with
what tolerance; the after-hours session ships disabled. A level only counts
for bars that start after it exists (pre-market levels from 09:30, opening
range from 10:00, today's regular-session levels from 16:00), so a bar can
never "touch" a level that it helped to create.

The state dict is JSON-serialisable; check_levels never mutates the one you
pass in -- commit the returned one only after the alert was delivered.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import time as dt_time

import pandas as pd

import config
from levels import (
    OPENING_RANGE_END,
    PREMARKET_READY,
    REGULAR_END,
    REGULAR_START,
    SupportResistance,
    pivot_point,
)

MIDNIGHT = dt_time(0, 0)

# First time of day (ET, bar start) from which a bar may touch each level.
VALID_FROM = {
    "PMH": PREMARKET_READY, "PML": PREMARKET_READY,
    "ORH": OPENING_RANGE_END, "ORL": OPENING_RANGE_END,
    "TDH": REGULAR_END, "TDL": REGULAR_END, "TDC": REGULAR_END,
}
DYNAMIC = {"DH", "DL"}   # running day extremes: "touching" them is meaningless


# --------------------------------------------------------------------------
# Settings / data containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionRule:
    name: str
    start: dt_time
    end: dt_time
    families: tuple
    tol_pct: float
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "SessionRule":
        return cls(d["name"], d["start"], d["end"], tuple(d["families"]),
                   float(d["tol_pct"]), bool(d.get("enabled", True)))

    def contains(self, t: dt_time) -> bool:
        return self.start <= t < self.end


@dataclass
class LevelAlertSettings:
    enabled: bool = config.LEVEL_ALERTS_ENABLED
    base_rule: str = config.BASE_RULE
    max_age_min: float = config.LEVEL_MAX_AGE_MIN
    catchup_min: float = config.LEVEL_CATCHUP_MIN
    rearm_min: float = config.LEVEL_REARM_MIN
    max_alerts_per_day: int = config.LEVEL_MAX_ALERTS_PER_DAY
    sessions: list = field(
        default_factory=lambda: [SessionRule.from_dict(d) for d in config.LEVEL_SESSIONS]
    )

    def session_for(self, t: dt_time) -> SessionRule | None:
        for rule in self.sessions:
            if rule.enabled and rule.contains(t):
                return rule
        return None


@dataclass(frozen=True)
class LevelEvent:
    symbol: str
    level: str            # "PDH", "ORL", ...
    value: float
    session: str
    bar_time: pd.Timestamp   # START of the bar that touched
    open: float
    high: float
    low: float
    close: float
    approach: str         # "from below" / "from above" / "at the level"


# --------------------------------------------------------------------------
# Level collection
# --------------------------------------------------------------------------
def collect_levels(df: pd.DataFrame, sr: SupportResistance,
                   now: pd.Timestamp) -> dict[str, tuple[float, dt_time]]:
    """{name: (value, valid_from)} of every static level available at `now`:
    the families levels.build_levels labelled, plus R1/S1, plus today's
    regular-session TDH/TDL/TDC once 16:00 has passed."""
    values = {k: v for k, v in sr.labeled().items() if k not in DYNAMIC}

    prev = sr.previous_day
    piv = pivot_point(prev.high, prev.low, prev.close)
    values["R1"], values["S1"] = piv["R1"], piv["S1"]

    if now.time() >= REGULAR_END:
        mask = (df.index.date == sr.day) & (df.index.time >= REGULAR_START) \
               & (df.index.time < REGULAR_END)
        reg = df.loc[mask]
        if not reg.empty:
            values["TDH"] = round(float(reg["high"].max()), 2)
            values["TDL"] = round(float(reg["low"].min()), 2)
            values["TDC"] = round(float(reg["close"].iloc[-1]), 2)

    return {name: (val, VALID_FROM.get(name, MIDNIGHT)) for name, val in values.items()}


# --------------------------------------------------------------------------
# Touch detection
# --------------------------------------------------------------------------
def check_levels(
    symbol: str,
    df: pd.DataFrame,
    sr: SupportResistance | None,
    now: pd.Timestamp,
    state: dict | None = None,
    settings: LevelAlertSettings | None = None,
) -> tuple[list[LevelEvent], dict]:
    """Return (new touch events, updated state). `df` = closed 5m bars
    (open/high/low/close, US/Eastern index); `sr` = levels.build_levels(...)
    as of `now`. The input state is never mutated."""
    s = settings or LevelAlertSettings()
    if not s.enabled or df is None or df.empty or sr is None:
        return [], (state or {})

    step = pd.Timedelta(s.base_rule)
    day = str(now.date())
    st = copy.deepcopy(state) if state else {}
    if st.get("day") != day:                       # new trading day: forget yesterday's alerts
        st = {"day": day, "touched": {}, "count": 0, "cursor": st.get("cursor")}
    st.setdefault("touched", {})
    st.setdefault("count", 0)

    closed = df.loc[(df.index + step) <= now]
    if closed.empty:
        return [], st

    cursor = st.get("cursor")
    if cursor:
        since = max(pd.Timestamp(cursor), now - pd.Timedelta(minutes=s.catchup_min))
    else:
        since = now - pd.Timedelta(minutes=s.max_age_min)
    new_bars = closed.loc[(closed.index + step) > since]

    levels = collect_levels(df, sr, now)
    prev_close = df["close"].shift(1)
    contiguous = df.index.to_series().diff() == step     # previous bar is the one right before

    events: list[LevelEvent] = []
    for ts, bar in new_bars.iterrows():
        rule = s.session_for(ts.time())
        if rule is None:
            continue
        pc = prev_close.loc[ts] if contiguous.loc[ts] else float("nan")
        has_pc = pc == pc                                   # not NaN
        low = min(bar["low"], pc) if has_pc else bar["low"]   # path = bar range + gap from prev close
        high = max(bar["high"], pc) if has_pc else bar["high"]
        ref = pc if has_pc else bar["open"]
        bar_end = ts + step

        for name in rule.families:
            if name not in levels:
                continue
            value, valid_from = levels[name]
            if ts.time() < valid_from:
                continue
            tol = value * rule.tol_pct / 100.0
            if not (low <= value + tol and high >= value - tol):
                continue

            key = f"{name}|{rule.name}"
            last = st["touched"].get(key)
            if last is not None:
                if s.rearm_min <= 0 or (bar_end - pd.Timestamp(last)) < pd.Timedelta(minutes=s.rearm_min):
                    continue
            if st["count"] >= s.max_alerts_per_day:
                continue

            approach = ("from below" if ref < value - tol
                        else "from above" if ref > value + tol else "at the level")
            events.append(LevelEvent(symbol, name, value, rule.name, ts,
                                     float(bar["open"]), float(bar["high"]),
                                     float(bar["low"]), float(bar["close"]), approach))
            st["touched"][key] = bar_end.isoformat()
            st["count"] += 1

    st["cursor"] = (closed.index[-1] + step).isoformat()
    return events, st


# --------------------------------------------------------------------------
# Message text (no '+', '&' or '#': the Telegram URL is not percent-encoded)
# --------------------------------------------------------------------------
def format_events(symbol: str, events: list[LevelEvent]) -> str:
    """One line per bar:
    'SPY level touch (11:40): PDH 602.10 from below, R1 602.15 from below | bar H 602.12 L 601.80 C 601.95'"""
    by_bar: dict[pd.Timestamp, list[LevelEvent]] = {}
    for e in events:
        by_bar.setdefault(e.bar_time, []).append(e)

    lines = []
    for ts, evs in sorted(by_bar.items()):
        tag = "" if evs[0].session == "regular" else f" [{evs[0].session}]"
        touched = ", ".join(f"{e.level} {e.value:.2f} {e.approach}" for e in evs)
        b = evs[0]
        lines.append(f"{symbol} level touch ({ts:%H:%M}){tag}: {touched} | "
                     f"bar H {b.high:.2f} L {b.low:.2f} C {b.close:.2f}")
    return "\n".join(lines)
