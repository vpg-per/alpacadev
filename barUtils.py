"""Pure-pandas bar helpers (no network, no Alpaca). Safe to import anywhere."""
from __future__ import annotations

import pandas as pd

OHLC = ["open", "high", "low", "close"]


def round_ohlc(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Return a copy with whichever of open/high/low/close exist rounded."""
    out = df.copy()
    cols = [c for c in OHLC if c in out.columns]
    if cols:
        out[cols] = out[cols].round(decimals)
    return out


def drop_incomplete_last_candle(
    df: pd.DataFrame, rule: str = "5min", now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Drop the last row if its candle window hasn't elapsed yet (still
    forming). E.g. a 5-min bar stamped 10:15 covers 10:15-10:20; at 10:17 it
    is dropped. `now` is injectable for tests/backtests."""
    if df.empty:
        return df
    now = now if now is not None else pd.Timestamp.now(tz=df.index.tz)
    if df.index[-1] + pd.Timedelta(rule) > now:
        return df.iloc[:-1]
    return df


def resample_bars(
    df: pd.DataFrame,
    rule: str,
    base_rule: str = "5min",
    drop_incomplete: bool = False,
) -> pd.DataFrame:
    """Resample OHLC(V) bars to a larger timeframe ('15min', '1h', ...).

    drop_incomplete=True removes the final bucket unless the *data itself*
    reaches the end of that bucket (last base bar + base_rule >= bucket end).
    This avoids treating a 15m bucket built from 1-2 delayed 5m bars as
    closed, and is independent of wall-clock time. Interior gaps (thin
    overnight sessions) do not cause a bucket to be dropped.
    """
    ohlcv = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    agg = {c: fn for c, fn in ohlcv.items() if c in df.columns}

    resampled = df.resample(rule).agg(agg).dropna(subset=["open"])
    resampled["interval"] = rule
    if "symbol" in df.columns and not df.empty:
        resampled["symbol"] = df["symbol"].iloc[0]

    if drop_incomplete and not resampled.empty:
        bucket_end = resampled.index[-1] + pd.Timedelta(rule)
        if df.index[-1] + pd.Timedelta(base_rule) < bucket_end:
            resampled = resampled.iloc[:-1]

    return round_ohlc(resampled)


def resample_many(
    df: pd.DataFrame,
    rules,
    base_rule: str = "5min",
    drop_incomplete: bool = True,
) -> dict[str, pd.DataFrame]:
    """Resample one base frame to several timeframes: {rule: frame}."""
    return {r: resample_bars(df, r, base_rule, drop_incomplete) for r in rules}


def market_closed_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True where a US/Eastern timestamp falls in the weekly closure
    (Fri 20:00 -> Sun 20:00). Holidays are not modelled."""
    dow, hour = index.dayofweek, index.hour
    return (dow == 5) | ((dow == 4) & (hour >= 20)) | ((dow == 6) & (hour < 20))


def find_missing_periods(
    df: pd.DataFrame,
    freq: str = "5min",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    exclude_closed: bool = True,
) -> pd.DatetimeIndex:
    """Timestamps missing from df's index over [start, end], assuming
    continuous `freq` bars. exclude_closed skips the weekend closure so it
    doesn't inflate the count (index must be US/Eastern for that to be right)."""
    if df.empty:
        raise ValueError("df is empty -- nothing to check.")
    start = start or df.index.min()
    end = end or df.index.max()
    expected = pd.date_range(start=start, end=end, freq=freq, tz=df.index.tz)
    if exclude_closed:
        expected = expected[~market_closed_mask(expected)]
    return expected.difference(df.index)


def report_missing_periods(
    df: pd.DataFrame,
    freq: str = "5min",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    exclude_closed: bool = True,
) -> pd.DataFrame:
    """Group consecutive missing timestamps into gaps:
    columns gap_start, gap_end, bars_missing."""
    missing = find_missing_periods(df, freq, start, end, exclude_closed)
    if missing.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "bars_missing"])

    step = pd.Timedelta(freq)
    s = missing.to_series()
    run_id = (s.diff() != step).cumsum()
    grouped = s.groupby(run_id)
    gaps = grouped.agg(gap_start="min", gap_end="max")
    gaps["bars_missing"] = grouped.size().values
    return gaps.reset_index(drop=True)
