"""MACD-momentum analysis across 5m / 15m / 30m. Network-free at its core:

    SignalAnalyzer(settings).analyze_frame(symbol, df_5m, now=None)   # pure, testable
    SignalAnalyzer(settings).analyze(symbol, now=None)                # fetches via Alpaca

`now` is the "as of" time (tz-aware). Leave it None for live use; pass a past
timestamp to replay what the analyzer would have seen then (debugging).

Default alert rules (config.py):
  Bullish: 5m MACD changed from Down/Neutral to Up, AND 5m EMA9 > EMA21,
           AND 15m MACD is Up or Neutral, AND 30m MACD is Up or Neutral.
  Bearish: 5m MACD changed from Up/Neutral to Down, AND 5m EMA9 < EMA21,
           AND 15m MACD is Down or Neutral, AND 30m MACD is Down or Neutral.
The flip bar must also be fresh (config.MAX_SIGNAL_AGE_MIN).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping

import pandas as pd

import config
from barUtils import drop_incomplete_last_candle, resample_many
from indicators import add_ema, add_macd, apply_indicators

log = logging.getLogger(__name__)


@dataclass
class AnalyzerSettings:
    base_rule: str = config.BASE_RULE
    timeframes: list = field(default_factory=lambda: list(config.TIMEFRAMES))
    confirm_rules: list = field(default_factory=lambda: list(config.CONFIRM_RULES))
    macd_fast: int = config.MACD_FAST
    macd_slow: int = config.MACD_SLOW
    macd_signal: int = config.MACD_SIGNAL
    ema_periods: tuple = (config.EMA_FAST, config.EMA_SLOW)   # (fast, slow), base timeframe only
    ema_filter: bool = config.EMA_FILTER
    min_bars: int = config.MIN_BARS
    max_signal_age_min: float = config.MAX_SIGNAL_AGE_MIN
    htf_confirm: Mapping[str, tuple] = field(default_factory=lambda: dict(config.HTF_CONFIRM))
    # {"5min": [...], "15min": [...], "*": [...]} of (name, params); see config.EXTRA_INDICATORS
    extra_indicators: Mapping[str, list] = field(
        default_factory=lambda: {k: list(v) for k, v in config.EXTRA_INDICATORS.items()}
    )

    def __post_init__(self):
        unknown = set(self.confirm_rules) - set(self.timeframes)
        if unknown:
            raise ValueError(f"confirm_rules {sorted(unknown)} not in timeframes {self.timeframes}")
        if len(self.ema_periods) != 2:
            raise ValueError("ema_periods must be (fast, slow)")
        if isinstance(self.extra_indicators, (list, tuple)):  # legacy: list applies to all timeframes
            self.extra_indicators = {"*": list(self.extra_indicators)}

    @property
    def ema_cols(self) -> tuple[str, str]:
        return tuple(f"ema_{p}" for p in self.ema_periods)


@dataclass
class AnalysisResult:
    symbol: str
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)  # {"5min": df, "15min": df, ...}
    signal_row: pd.Series | None = None         # latest 5m row, set only when alertable
    htf_rows: dict[str, pd.Series] = field(default_factory=dict)   # latest row per higher timeframe
    ema_cols: tuple = ()                        # names of the (fast, slow) EMA columns on the 5m frame
    reason: str = ""                            # why there is no alert (for logging)

    @property
    def has_signal(self) -> bool:
        return self.signal_row is not None

    @property
    def direction(self) -> str | None:
        return None if self.signal_row is None else self.signal_row["MACDTrend"]


def trend_flip(row: pd.Series) -> str | None:
    """Up (bullish) if the bar's MACD trend just changed from Down/Neutral to
    Up; Down (bearish) if it just changed from Up/Neutral to Down; else None."""
    trend, cross = row["MACDTrend"], row["macdtrend_cross"]
    if trend == "Up" and cross > 0:
        return "Up"
    if trend == "Down" and cross < 0:
        return "Down"
    return None


def is_signal_fresh(
    bar_start: pd.Timestamp,
    base_rule: str = "5min",
    max_age_min: float = 10,
    now: pd.Timestamp | None = None,
) -> bool:
    """True if the bar closed within the last `max_age_min` minutes."""
    now = now if now is not None else pd.Timestamp.now(tz=bar_start.tz)
    bar_end = bar_start + pd.Timedelta(base_rule)
    return (now - bar_end) <= pd.Timedelta(minutes=max_age_min)


class SignalAnalyzer:
    """Stateless between calls: results are returned, nothing is cached on
    the instance, so one instance can safely handle many symbols."""

    def __init__(
        self,
        settings: AnalyzerSettings | None = None,
        fetcher: Callable[..., pd.DataFrame] | None = None,
    ):
        """fetcher(symbol, interval) -> DataFrame. If you pass `now` to
        analyze(), it is called as fetcher(symbol, interval, end=now)."""
        self.settings = settings or AnalyzerSettings()
        self._fetcher = fetcher

    def analyze(self, symbol: str, now: pd.Timestamp | None = None) -> AnalysisResult:
        fetcher = self._fetcher
        if fetcher is None:
            from alaDataManager import fetch_stock_data  # lazy: keeps Alpaca optional
            fetcher = fetch_stock_data
        kwargs = {} if now is None else {"end": now}
        return self.analyze_frame(symbol, fetcher(symbol, self.settings.base_rule, **kwargs), now)

    def analyze_frame(
        self, symbol: str, data_base: pd.DataFrame, now: pd.Timestamp | None = None
    ) -> AnalysisResult:
        s = self.settings
        n = 0 if data_base is None else len(data_base)
        if n < s.min_bars:
            return AnalysisResult(symbol, reason=f"insufficient data ({n}/{s.min_bars} bars)")

        data_base = drop_incomplete_last_candle(data_base, s.base_rule, now)
        higher = resample_many(data_base, s.timeframes, s.base_rule, drop_incomplete=True)

        frames = {s.base_rule: self._prepare(data_base, s.base_rule)}
        for rule, df in higher.items():
            frames[rule] = self._prepare(df, rule) if not df.empty else df

        result = AnalysisResult(symbol, frames, ema_cols=s.ema_cols)
        last = frames[s.base_rule].iloc[-1]
        direction = trend_flip(last)

        if direction is None:
            result.reason = f"no {s.base_rule} MACD flip on latest bar"
            return result
        if not is_signal_fresh(last.name, s.base_rule, s.max_signal_age_min, now):
            result.reason = "flip rejected: stale"
            return result

        if s.ema_filter:
            fast_col, slow_col = s.ema_cols
            fast, slow = last[fast_col], last[slow_col]
            if not (fast > slow if direction == "Up" else fast < slow):
                result.reason = f"{direction} flip rejected: {fast_col} {fast} vs {slow_col} {slow}"
                return result

        for rule in s.confirm_rules:
            df = frames[rule]
            if df.empty:
                result.reason = f"flip rejected: no {rule} data"
                return result
            trend = df["MACDTrend"].iloc[-1]
            if trend not in s.htf_confirm[direction]:
                result.reason = f"{direction} flip rejected: {rule} MACD is {trend}"
                return result

        result.signal_row = last
        result.htf_rows = {r: frames[r].iloc[-1] for r in s.timeframes if not frames[r].empty}
        return result

    def _prepare(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        s = self.settings
        out = add_macd(df, s.macd_fast, s.macd_slow, s.macd_signal)
        if rule == s.base_rule:
            out = add_ema(out, s.ema_periods)
        specs = list(s.extra_indicators.get("*", [])) + list(s.extra_indicators.get(rule, []))
        if specs:
            out = apply_indicators(out, specs)
        return out
