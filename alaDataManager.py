"""Alpaca API client setup."""
from __future__ import annotations

import os
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import config

load_dotenv()


def get_client() -> StockHistoricalDataClient:
    """Build an authenticated Alpaca historical data client from env vars
    ALA_API_KEY / ALA_SECRET_KEY (set these in a .env file)."""
    api_key = os.getenv("ALA_API_KEY")
    secret_key = os.getenv("ALA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALA_API_KEY / ALA_SECRET_KEY. Set them in your .env file."
        )
    return StockHistoricalDataClient(api_key, secret_key)


def download_stock_data(
    symbol: str,
    startPeriod,
    endPeriod,
    interval: str = "5m",
) -> pd.DataFrame | None:
    """Fetch OHLC bars directly from Yahoo Finance's chart API (ported from
    dataManager.ServiceManager.download_stock_data). Used here as a
    near-real-time supplement to Alpaca, whose feeds can lag by 15+ minutes.

    startPeriod/endPeriod may be datetimes/Timestamps or raw unix seconds.
    Returns a DataFrame indexed by tz-aware US/Eastern timestamps, or None
    on failure / no data.
    """
    period1 = int(startPeriod.timestamp()) if hasattr(startPeriod, "timestamp") else int(startPeriod)
    period2 = int(endPeriod.timestamp()) if hasattr(endPeriod, "timestamp") else int(endPeriod)

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": interval,
        "includePrePost": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        quotes = result["indicators"]["quote"][0]

        ts_arr = np.asarray(result["timestamp"], dtype="int64")
        df = pd.DataFrame(
            {
                "open": np.round(np.asarray(quotes["open"], dtype="float64"), 2),
                "high": np.round(np.asarray(quotes["high"], dtype="float64"), 2),
                "low": np.round(np.asarray(quotes["low"], dtype="float64"), 2),
                "close": np.round(np.asarray(quotes["close"], dtype="float64"), 2),
            },
            index=pd.to_datetime(ts_arr, unit="s", utc=True),
        )
        df.index.name = "timestamp"
        df.dropna(inplace=True)
        if df.empty:
            return None

        df.index = df.index.tz_convert(config.EASTERN)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Yahoo data for {symbol}: {e}")
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing Yahoo data for {symbol}: {e}")
    return None


def fetch_recent_yahoo(
    symbol: str = config.SYMBOL,
    minutes: int = 30,
) -> pd.DataFrame | None:
    now_et = datetime.now(config.EASTERN)
    start = now_et - timedelta(minutes=minutes)

    df = download_stock_data(symbol, start, now_et, interval="5m")
    if df is None or df.empty:
        return None

    mask = (df.index.time >= dt_time(4, 0)) & (df.index.time < dt_time(20, 0))
    df = df.loc[mask].copy()
    if df.empty:
        return None
    
    df = drop_incomplete_last_candle(df, "5min")
    if df.empty:
        return None

    df["feed"] = "yahoo"
    df = round_ohlc(df)
    return df

def drop_incomplete_last_candle(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """Drop the last row of df if its candle window hasn't fully elapsed
    yet (i.e. it's still forming, not a closed bar). E.g. for a 5-min
    candle stamped 10:15, the window is 10:15-10:20 -- if it's only
    10:17 right now, that candle isn't done printing yet.
    """
    if df.empty:
        return df

    freq = pd.Timedelta(rule)
    now = pd.Timestamp.now(tz=df.index.tz)
    last_ts = df.index[-1]

    if last_ts + freq > now:
        df = df.iloc[:-1]

    return df

def append_recent_yahoo_data(
    df: pd.DataFrame,
    symbol: str = config.SYMBOL,
    minutes: int = 30,
) -> pd.DataFrame:
    """Patch the tail of an Alpaca-sourced, 5-min-bar `df` with fresher
    5-min bars rebuilt from Yahoo Finance's near-real-time 1-min data, to
    work around Alpaca feed delays. On any overlapping timestamp, the
    Yahoo bar wins since it's the more current one.

    No-op (returns df unchanged) if Yahoo has no data for the window --
    e.g. overnight hours, which Yahoo doesn't cover at all.
    """
    recent = fetch_recent_yahoo(symbol, minutes)
    if recent is None or recent.empty:
        return df

    recent_5m = resample_bars(recent, "5min")
    if recent_5m.empty:
        return df
    recent_5m["symbol"] = symbol

    merged = pd.concat([df, recent_5m])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged = round_ohlc(merged)
    return merged


def to_eastern(df: pd.DataFrame, symbol: str = config.SYMBOL) -> pd.DataFrame:
    """Flatten Alpaca's (symbol, timestamp) MultiIndex to a plain timestamp
    index and convert it from UTC to US Eastern time."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol").copy()
    else:
        df = df.copy()
    df.index = df.index.tz_convert(config.EASTERN)
    return df

def round_ohlc(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round open/high/low/close columns (whichever are present) to the
    given number of decimal places."""
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if cols:
        df[cols] = df[cols].round(decimals)
    return df

def fetch_feed(
    client: StockHistoricalDataClient,
    feed: str,
    start: datetime,
    symbol: str = config.SYMBOL,
    timeframe: TimeFrame = config.BASE_TIMEFRAME,
) -> pd.DataFrame | None:
    """Fetch bars for a single feed. Returns None (instead of raising) if
    the feed/subscription isn't available on this account."""
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=start,
        feed=feed,
    )
    try:
        bars = client.get_stock_bars(request).df
    except Exception as e:
        print(f"Feed '{feed}' unavailable (needs a matching subscription?): {e}")
        return None

    if bars.empty:
        return None

    df = to_eastern(bars, symbol)
    df["feed"] = feed
    df = df.drop(columns=["vwap","volume","trade_count","feed"], errors="ignore")
    return df


def fetch_combined(
    client: StockHistoricalDataClient,
    feeds: list[str],
    start: datetime,
    symbol: str = config.SYMBOL,
    timeframe: TimeFrame = config.BASE_TIMEFRAME,
) -> pd.DataFrame:
    """Fetch each feed and merge into one gap-filled, deduplicated timeline.

    Feeds are merged in the order given, so earlier feeds in `feeds` win on
    any overlapping timestamps (e.g. put "sip" before "boats" since sip is
    the superset for regular/extended hours).
    """
    frames = [fetch_feed(client, feed, start, symbol, timeframe) for feed in feeds]
    frames = [f for f in frames if f is not None]

    if not frames:
        raise RuntimeError(
            "No feeds returned data -- check your subscriptions and symbol."
        )

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined["interval"] = "5min"
    combined["symbol"]=symbol    
    combined = round_ohlc(combined)

    # Alpaca feeds can lag; patch in the freshest ~30 min from Yahoo
    # Finance (pre/regular/post market only -- no overnight coverage there).
    combined = append_recent_yahoo_data(combined, symbol)
    combined = drop_incomplete_last_candle(combined, "5min")
    
    return combined

def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLC bar data to a larger timeframe (e.g. '15min', '1h').

    df must have a DatetimeIndex and open/high/low/close columns.
    """
    ohlc_agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    agg = {col: fn for col, fn in ohlc_agg.items() if col in df.columns}

    resampled = df.resample(rule).agg(agg).dropna(subset=["open"])
    resampled["interval"] = rule    
    if "symbol" in df.columns and not df.empty:
        resampled["symbol"] = df["symbol"].iloc[0]
    resampled = round_ohlc(resampled)

    return resampled

def fetch_stock_data(symbol, interval) -> pd.DataFrame:
    client = get_client()
    start = config.EASTERN.localize(datetime.now() - timedelta(days=config.LOOKBACK_DAYS))

    combined = fetch_combined(client, config.FEEDS, start, symbol)
    if interval == "5min":
        missing = find_missing_periods(combined)  
        print(f"Missing count: { missing.size }")
        return combined
    
    resampled = resample_bars(combined, interval)
    return resampled

def find_missing_periods(
    df: pd.DataFrame,
    freq: str = "5min",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Return the timestamps missing from df's index over [start, end],
    assuming continuous `freq`-spaced bars are expected (e.g. sip + overnight
    feeds combined should give an unbroken 5-min timeline).

    If start/end are omitted, uses df.index.min()/max().
    """
    if df.empty:
        raise ValueError("df is empty -- nothing to check.")

    start = start or df.index.min()
    end   = end or df.index.max()

    expected = pd.date_range(start=start, end=end, freq=freq, tz=df.index.tz)
    missing  = expected.difference(df.index)
    return missing


def report_missing_periods(
    df: pd.DataFrame,
    freq: str = "5min",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Like find_missing_periods, but also groups consecutive missing
    timestamps into contiguous gap ranges for a more readable report.
    Returns a DataFrame with columns: gap_start, gap_end, bars_missing.
    """
    missing = find_missing_periods(df, freq, start, end)
    if missing.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "bars_missing"])

    step = pd.Timedelta(freq)
    # Identify breaks between consecutive missing timestamps to group runs.
    is_new_run = missing.to_series().diff() != step
    run_id = is_new_run.cumsum()

    grouped = missing.to_series().groupby(run_id)
    gaps = grouped.agg(gap_start="min", gap_end="max")
    gaps["bars_missing"] = grouped.size().values
    return gaps.reset_index(drop=True)