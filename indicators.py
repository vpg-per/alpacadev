"""Pure `DataFrame -> DataFrame` indicators (no I/O, no config, no mutation
of the input). Add a new one by decorating it with @register_indicator; it
can then be enabled via config.EXTRA_INDICATORS or apply_indicators().

Conventions: each function takes a frame with open/high/low/close[/volume],
returns a copy with new columns appended.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

INDICATORS: dict[str, Callable[..., pd.DataFrame]] = {}

_RANK = {"Down": -1, "Neutral": 0, "Up": 1}


def register_indicator(name: str):
    def deco(fn):
        INDICATORS[name] = fn
        return fn
    return deco


def apply_indicators(df: pd.DataFrame, specs: Iterable) -> pd.DataFrame:
    """Apply a list of indicators. Each spec is "name" or ("name", {params})."""
    out = df
    for spec in specs:
        name, params = (spec, {}) if isinstance(spec, str) else (spec[0], spec[1] or {})
        if name not in INDICATORS:
            raise KeyError(f"Unknown indicator '{name}'. Registered: {sorted(INDICATORS)}")
        out = INDICATORS[name](out, **params)
    return out


def trend_cross(trend: pd.Series) -> pd.Series:
    """+1 when a Down/Neutral/Up label ranks higher than the previous bar,
    -1 when lower, else 0."""
    rank = trend.map(_RANK)
    delta = rank - rank.shift(1)
    return pd.Series(
        np.select([delta > 0, delta < 0], [1, -1], default=0), index=trend.index
    ).astype("int8")


@register_indicator("macd")
def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
    keep_lines: bool = False,
) -> pd.DataFrame:
    """Adds MACDTrend (Up / Down / Neutral) and macdtrend_cross (-1/0/1).
    Up = MACD above signal AND above zero; Down = below both; else Neutral.
    keep_lines=True also keeps macd, msignal, histogram."""
    out = df.copy()
    px = out[price_col].astype("float64")
    macd_line = px.ewm(span=fast, adjust=False).mean() - px.ewm(span=slow, adjust=False).mean()
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()

    up = (macd_line > sig_line) & (macd_line > 0)
    down = (macd_line < sig_line) & (macd_line < 0)
    out["MACDTrend"] = pd.Series(
        np.select([up, down], ["Up", "Down"], default="Neutral"), index=out.index
    )
    out["macdtrend_cross"] = trend_cross(out["MACDTrend"])

    if keep_lines:
        out["macd"] = macd_line.round(4)
        out["msignal"] = sig_line.round(4)
        out["histogram"] = (macd_line - sig_line).round(4)
    return out


@register_indicator("ema")
def add_ema(
    df: pd.DataFrame, periods: Iterable[int] = (9, 21), price_col: str = "close"
) -> pd.DataFrame:
    """Adds ema_<period> for each period."""
    out = df.copy()
    px = out[price_col].astype("float64")
    for p in periods:
        out[f"ema_{p}"] = px.ewm(span=p, adjust=False).mean().round(2)
    return out


@register_indicator("vwap")
def add_vwap(df: pd.DataFrame, session_start_hour: int = 4) -> pd.DataFrame:
    """Adds session-anchored `vwap` (typical price * volume). The accumulator
    resets at `session_start_hour` (US/Eastern index): 4 = pre-market open,
    9 would anchor to roughly the regular session, 20 to the overnight open.
    Requires a `volume` column."""
    if "volume" not in df.columns:
        raise ValueError("vwap needs a 'volume' column")
    out = df.copy()
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    vol = out["volume"].astype("float64")
    session = (out.index - pd.Timedelta(hours=session_start_hour)).normalize()
    cum_pv = (typical * vol).groupby(session).cumsum()
    cum_v = vol.groupby(session).cumsum()
    out["vwap"] = (cum_pv / cum_v.replace(0, np.nan)).round(2)
    return out
