"""Market-data access: Alpaca historical bars (sip + overnight) with a Yahoo
Finance near-real-time supplement. Pure bar helpers live in barUtils.py and
are re-exported here for backward compatibility.

Reuse notes: every function takes explicit arguments and only falls back to
config.py when an argument is None. Credentials are read lazily from
ALA_API_KEY / ALA_SECRET_KEY (or passed to get_client()).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from barUtils import (  # noqa: F401  (re-exported)
    OHLC,
    drop_incomplete_last_candle,
    find_missing_periods,
    report_missing_periods,
    resample_bars,
    resample_many,
    round_ohlc,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------
def now_eastern() -> datetime:
    """Current time as a tz-aware US/Eastern datetime (never naive)."""
    return datetime.now(config.EASTERN)


def in_extended_hours(ts: datetime | pd.Timestamp | None = None) -> bool:
    """True when ts (default: now, Eastern) is inside the Yahoo-covered window."""
    ts = ts or now_eastern()
    return config.EXTENDED_START <= ts.time() < config.EXTENDED_END


def _extended_hours_mask(index: pd.DatetimeIndex):
    return (index.time >= config.EXTENDED_START) & (index.time < config.EXTENDED_END)


def _to_epoch(value) -> int:
    return int(value.timestamp()) if hasattr(value, "timestamp") else int(value)


# --------------------------------------------------------------------------
# Alpaca
# --------------------------------------------------------------------------
def get_client(api_key: str | None = None, secret_key: str | None = None) -> StockHistoricalDataClient:
    """Authenticated Alpaca client. Falls back to env vars ALA_API_KEY /
    ALA_SECRET_KEY (a .env file is loaded if python-dotenv is installed)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = api_key or os.getenv("ALA_API_KEY")
    secret_key = secret_key or os.getenv("ALA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Missing ALA_API_KEY / ALA_SECRET_KEY. Set them in your .env file.")
    return StockHistoricalDataClient(api_key, secret_key)


def to_eastern(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Flatten Alpaca's (symbol, timestamp) MultiIndex and convert to US/Eastern."""
    symbol = symbol or config.SYMBOL
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol").copy()
    else:
        df = df.copy()
    df.index = df.index.tz_convert(config.EASTERN)
    return df


def fetch_feed(
    client: StockHistoricalDataClient,
    feed: str,
    start: datetime,
    symbol: str | None = None,
    timeframe: TimeFrame | None = None,
    end: datetime | None = None,
) -> pd.DataFrame | None:
    """Bars for one feed (up to `end` if given), or None (not an exception)
    if the feed/subscription isn't available."""
    symbol = symbol or config.SYMBOL
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe or config.BASE_TIMEFRAME,
        start=start,
        end=end,
        feed=feed,
    )
    try:
        bars = client.get_stock_bars(request).df
    except Exception as e:
        log.warning("Feed '%s' unavailable (needs a matching subscription?): %s", feed, e)
        return None

    if bars.empty:
        return None
    df = to_eastern(bars, symbol)
    return df.drop(columns=config.ALPACA_DROP_COLUMNS, errors="ignore")


# --------------------------------------------------------------------------
# Yahoo
# --------------------------------------------------------------------------
def download_stock_data(
    symbol: str,
    startPeriod,
    endPeriod,
    interval: str = "5m",
    include_prepost: bool = True,
    timeout: int = 15,
) -> pd.DataFrame | None:
    """OHLCV bars from Yahoo's chart API. start/end: tz-aware datetimes /
    Timestamps or unix seconds. Returns a US/Eastern-indexed frame, or None
    on failure / no data."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": _to_epoch(startPeriod),
        "period2": _to_epoch(endPeriod),
        "interval": interval,
        "includePrePost": str(include_prepost).lower(),
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()["chart"]["result"][0]
        if "timestamp" not in result:
            return None  # no bars in window (weekend / off-hours)

        quotes = result["indicators"]["quote"][0]
        ts_arr = np.asarray(result["timestamp"], dtype="int64")
        vol = quotes.get("volume") or [0] * len(ts_arr)
        df = pd.DataFrame(
            {
                "open": np.asarray(quotes["open"], dtype="float64"),
                "high": np.asarray(quotes["high"], dtype="float64"),
                "low": np.asarray(quotes["low"], dtype="float64"),
                "close": np.asarray(quotes["close"], dtype="float64"),
                "volume": np.asarray(vol, dtype="float64"),
            },
            index=pd.to_datetime(ts_arr, unit="s", utc=True),
        )
        df.index.name = "timestamp"
        df = df.dropna(subset=OHLC)
        if df.empty:
            return None
        df["volume"] = df["volume"].fillna(0)
        df.index = df.index.tz_convert(config.EASTERN)
        return round_ohlc(df)

    except requests.exceptions.RequestException as e:
        log.warning("Error fetching Yahoo data for %s: %s", symbol, e)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        log.warning("Error parsing Yahoo data for %s: %s", symbol, e)
    return None


def fetch_sector_data_yahoo(
    symbol: str,
    days_back: int | None = None,
    end_et: datetime | None = None,
) -> pd.DataFrame | None:
    """30-min extended-hours bars from Yahoo (used by sector-comparison code)."""
    days_back = config.LOOKBACK_DAYS if days_back is None else days_back
    end_et = end_et or now_eastern()
    df = download_stock_data(symbol, end_et - timedelta(days=days_back), end_et, interval="30m")
    if df is None or df.empty:
        return None
    df = df.loc[_extended_hours_mask(df.index)].copy()
    if df.empty:
        return None
    df["feed"] = "yahoo"
    df["symbol"] = symbol
    return round_ohlc(df)


def fetch_recent_yahoo(
    symbol: str | None = None,
    minutes: int | None = None,
    interval: str = "5m",
    rule: str = "5min",
) -> pd.DataFrame | None:
    """Last `minutes` of closed Yahoo bars (still-forming bar removed)."""
    symbol = symbol or config.SYMBOL
    minutes = minutes or config.YAHOO_RECENT_MINUTES
    now = now_eastern()
    df = download_stock_data(symbol, now - timedelta(minutes=minutes), now, interval=interval)
    if df is None or df.empty:
        return None
    df = df.loc[_extended_hours_mask(df.index)].copy()
    df = drop_incomplete_last_candle(df, rule)
    if df.empty:
        return None
    df["feed"] = "yahoo"
    return round_ohlc(df)


def append_recent_yahoo_data(
    df: pd.DataFrame,
    symbol: str | None = None,
    minutes: int | None = None,
) -> pd.DataFrame:
    """Patch the tail of an Alpaca 5-min frame with fresher Yahoo 5-min bars
    (Alpaca feeds can lag 15+ min). Yahoo wins on overlapping timestamps.
    No-op if Yahoo has nothing (overnight / weekend). Note Yahoo volume can
    differ from SIP consolidated volume on the patched bars."""
    symbol = symbol or config.SYMBOL
    recent = fetch_recent_yahoo(symbol, minutes)
    if recent is None or recent.empty:
        return df

    recent = recent[[c for c in df.columns if c in recent.columns]]
    merged = pd.concat([df, recent])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    for col in ("symbol", "interval"):
        if col in df.columns and not df.empty:
            merged[col] = df[col].iloc[-1]
    return round_ohlc(merged)


# --------------------------------------------------------------------------
# Combined fetch
# --------------------------------------------------------------------------
def fetch_combined(
    client: StockHistoricalDataClient,
    feeds: list[str],
    start: datetime,
    symbol: str | None = None,
    timeframe: TimeFrame | None = None,
    patch_with_yahoo: bool = True,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Merge feeds into one deduplicated 5-min timeline (earlier feeds win on
    overlap), optionally patch the tail from Yahoo, and drop the forming bar.

    `end` (tz-aware) turns this into an "as of" fetch for debugging: bars
    after `end` are excluded, "forming" is judged against `end` instead of
    the wall clock, and the Yahoo patch is skipped (it only knows "now")."""
    symbol = symbol or config.SYMBOL
    frames = [fetch_feed(client, f, start, symbol, timeframe, end) for f in feeds]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise RuntimeError("No feeds returned data -- check your subscriptions and symbol.")

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined = round_ohlc(combined)

    as_of = None
    if end is not None:
        as_of = pd.Timestamp(end)
        combined = combined.loc[combined.index <= as_of]
    elif patch_with_yahoo and in_extended_hours():
        combined = append_recent_yahoo_data(combined, symbol)
    combined = drop_incomplete_last_candle(combined, config.BASE_RULE, now=as_of)

    combined["interval"] = config.BASE_RULE
    combined["symbol"] = symbol
    return combined


def fetch_stock_data(
    symbol: str | None = None,
    interval: str = "5min",
    lookback_days: int | None = None,
    feeds: list[str] | None = None,
    client: StockHistoricalDataClient | None = None,
    patch_with_yahoo: bool = True,
    report_gaps: bool = False,
    end: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Closed bars for `symbol` at `interval` ('5min' returns base bars; any
    larger pandas rule is resampled with the forming bucket removed).

    end: optional tz-aware "as of" time (e.g. 2026-11-13 11:45 ET) for
    debugging; the lookback window then ends at `end` instead of now."""
    symbol = symbol or config.SYMBOL
    lookback_days = config.LOOKBACK_DAYS if lookback_days is None else lookback_days
    client = client or get_client()
    if end is not None:
        end = pd.Timestamp(end)
        if end.tzinfo is None:
            raise ValueError("end must be timezone-aware (US/Eastern recommended)")
        end = end.tz_convert(config.EASTERN).to_pydatetime()
    start = (end or now_eastern()) - timedelta(days=lookback_days)

    combined = fetch_combined(
        client, feeds or config.FEEDS, start, symbol,
        patch_with_yahoo=patch_with_yahoo, end=end,
    )
    if report_gaps:
        log.info("[%s] missing 5m bars: %d", symbol, len(find_missing_periods(combined)))

    if interval == config.BASE_RULE:
        return combined
    return resample_bars(combined, interval, base_rule=config.BASE_RULE, drop_incomplete=True)


def fetch_timeframes(
    symbol: str | None = None,
    rules: list[str] | None = None,
    lookback_days: int | None = None,
    feeds: list[str] | None = None,
    client: StockHistoricalDataClient | None = None,
    patch_with_yahoo: bool = True,
    end: datetime | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """One fetch, many timeframes: {"5min": ..., "15min": ..., "30min": ...}.
    `rules` defaults to config.TIMEFRAMES; the base 5-min frame is always included."""
    base = fetch_stock_data(
        symbol, config.BASE_RULE, lookback_days, feeds, client, patch_with_yahoo, end=end
    )
    out = {config.BASE_RULE: base}
    out.update(resample_many(base, rules or config.TIMEFRAMES, config.BASE_RULE))
    return out
