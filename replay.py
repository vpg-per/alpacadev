"""
Replay the alert rules over historical 5-minute bars to see HOW OFTEN they
fire and what price did afterwards. Nothing is sent to Telegram and no state
file is written.

Examples:
    python replay.py SPY --date 2026-11-13                 # one trading day
    python replay.py SPY GL IWM --date 2026-11-13 --days 5 # last 5 trading days up to 11/13
    python replay.py SPY --days 3 --compare                # current vs strict vs looser rule sets
    python replay.py SPY --days 5 --window 09:30-15:30     # only flips on bars starting 09:30-15:30 ET
    python replay.py SPY --days 5 --csv alerts.csv         # also save every alert to CSV

How it works (no look-ahead): for every 5m bar in the replay days it hands
the analyzer ONLY the data a live run would have had at that moment (the last
config.LOOKBACK_DAYS of bars up to that bar, "now" = bar close + --delay-min),
exactly like main.py does live. Forward results use the bar's close as the
entry price and are direction-adjusted (positive = price moved the way the
alert called). Forward windows that would cross a data gap (e.g. the 20:00
close) are skipped. This is a measurement of past behaviour, not a forecast,
and ignores spreads, slippage and commissions.

S/R levels (levels.py) are computed for every alert with the SAME time gates
a live run would have had at that moment (pre-market from 09:30, opening
range from 10:00, day high/low incl. pre-market from 11:45). The printed
table shows only the 2 nearest resistances and 2 nearest supports around
the alert's close (nearR/nearR2/nearS/nearS2, with %-distances); the --csv
file keeps the full labeled set (PDH/PDL/PDC/P/PMH/PML/ORH/ORL/DH/DL) for
deeper analysis.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import Counter
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd

import config
from dataAnalyzer import AnalyzerSettings, SignalAnalyzer
from levels import build_levels, distance_to_level, nearest_levels

log = logging.getLogger("replay")
BASE_STEP = pd.Timedelta(config.BASE_RULE)


# --------------------------------------------------------------------------
# Rule-set variants (--compare)
# --------------------------------------------------------------------------
def make_variants(compare: bool) -> dict[str, SignalAnalyzer]:
    variants = {"current": SignalAnalyzer(AnalyzerSettings())}
    if compare:
        strict = {"Up": ("Up",), "Down": ("Down",)}
        variants["strict"] = SignalAnalyzer(AnalyzerSettings(htf_confirm=strict))
        variants["5m flip + EMA only"] = SignalAnalyzer(AnalyzerSettings(confirm_rules=[]))
        variants["raw 5m MACD flips"] = SignalAnalyzer(
            AnalyzerSettings(confirm_rules=[], ema_filter=False)
        )
    return variants


# --------------------------------------------------------------------------
# Forward-looking measurements
# --------------------------------------------------------------------------
def forward(df: pd.DataFrame, pos: int, h: int) -> dict | None:
    """Long-side forward metrics from bar `pos` over the next h bars, or None
    if the window runs past the data or crosses a gap."""
    idx = df.index
    if pos + h >= len(df) or idx[pos + h] - idx[pos] != h * BASE_STEP:
        return None
    c0 = df["close"].iat[pos]
    win = df.iloc[pos + 1: pos + h + 1]
    return {
        "ret": (df["close"].iat[pos + h] / c0 - 1) * 100,
        "up": (win["high"].max() / c0 - 1) * 100,
        "down": (win["low"].min() / c0 - 1) * 100,
    }


def directional(f: dict, direction: str) -> dict:
    """Flip long-side metrics for a bearish alert. mae = worst move against."""
    if direction == "Up":
        return {"ret": f["ret"], "mfe": f["up"], "mae": -f["down"]}
    return {"ret": -f["ret"], "mfe": -f["down"], "mae": f["up"]}


# --------------------------------------------------------------------------
# Core replay (pure: no network)
# --------------------------------------------------------------------------
def norm_reason(reason: str) -> str:
    return re.sub(r"ema_\d+ [-\d.]+ vs ema_\d+ [-\d.]+", "EMA filter", reason)


def replay_frame(
    df: pd.DataFrame,
    symbol: str,
    analyzers: dict[str, SignalAnalyzer],
    replay_dates: set,
    window: tuple[dt_time, dt_time] | None = None,
    delay_min: float = 1,
    horizons: tuple[int, ...] = (6, 12, 24),
) -> dict:
    """Replay every 5m bar whose date is in replay_dates. Returns
    {"baseline": {h: [long returns]}, "variants": {name: record}}."""
    lookback = pd.Timedelta(days=config.LOOKBACK_DAYS)
    idx = df.index
    positions = [
        i for i, ts in enumerate(idx)
        if ts.date() in replay_dates and (window is None or window[0] <= ts.time() <= window[1])
    ]

    rec = {n: {"alerts": [], "evaluated": 0, "flips": 0, "reasons": Counter()} for n in analyzers}
    baseline = {h: [] for h in horizons}
    for i in positions:
        for h in horizons:
            f = forward(df, i, h)
            if f:
                baseline[h].append(f["ret"])

    for i in positions:
        ts = idx[i]
        now = ts + BASE_STEP + pd.Timedelta(minutes=delay_min)
        sub = df.loc[ts - lookback: ts]
        for name, an in analyzers.items():
            r = an.analyze_frame(symbol, sub, now)
            rr = rec[name]
            rr["evaluated"] += 1
            if r.reason.startswith("no ") and "MACD flip" in r.reason:
                continue
            if r.reason.startswith("insufficient"):
                rr["reasons"][r.reason.split(" (")[0]] += 1
                continue
            rr["flips"] += 1
            if not r.has_signal:
                rr["reasons"][norm_reason(r.reason)] += 1
                continue

            row = r.signal_row
            alert = {
                "symbol": symbol, "variant": name, "time": ts, "date": ts.date(),
                "direction": "Bullish" if r.direction == "Up" else "Bearish",
                "close": row["close"],
                r.ema_cols[0]: row[r.ema_cols[0]], r.ema_cols[1]: row[r.ema_cols[1]],
            }
            for rule, hrow in r.htf_rows.items():
                alert[rule] = hrow["MACDTrend"]
            # S/R levels exactly as a live run at `now` would have seen
            # them (time-gated: pre-market from 09:30, opening range from
            # 10:00, day extremes incl. pre-market from 11:45).
            try:
                sr = build_levels(sub, ts.date(), now)
            except Exception:
                sr = None
            if sr is not None:
                alert.update(sr.labeled())
                # 2 nearest resistances / supports around the alert's close:
                # nearR/nearS (nearest) and nearR2/nearS2 (second nearest),
                # each with its signed %-distance.
                near = nearest_levels(sr, row["close"], 2)
                for tag, lvls in (("R", near["resistance"]), ("S", near["support"])):
                    for i, lvl in enumerate(lvls, start=1):
                        col = f"near{tag}" if i == 1 else f"near{tag}{i}"
                        alert[col] = lvl
                        alert[f"{col}_pct"] = distance_to_level(row["close"], lvl)
            for h in horizons:
                f = forward(df, i, h)
                d = directional(f, r.direction) if f else None
                alert[f"ret_{h}"] = d["ret"] if d else np.nan
                alert[f"mae_{h}"] = d["mae"] if d else np.nan
                alert[f"mfe_{h}"] = d["mfe"] if d else np.nan
            rr["alerts"].append(alert)

    return {"baseline": baseline, "variants": rec, "bars": len(positions)}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _fmt_stats(vals) -> str:
    s = pd.Series(vals, dtype="float64").dropna()
    if s.empty:
        return "n=0"
    return (f"n={len(s):<3d} hit={100 * (s > 0).mean():5.1f}%  "
            f"mean={s.mean():+.3f}%  median={s.median():+.3f}%")


def print_report(symbol: str, result: dict, dates: list, horizons, show_alerts: bool) -> None:
    span = f"{dates[0]}" if len(dates) == 1 else f"{dates[0]} .. {dates[-1]} ({len(dates)} trading days)"
    for name, rr in result["variants"].items():
        alerts = pd.DataFrame(rr["alerts"])
        n = len(alerts)
        print(f"\n{'=' * 78}\n{symbol} | rule set: {name} | {span}\n{'=' * 78}")
        print(f"bars evaluated: {rr['evaluated']}   MACD flips seen: {rr['flips']}   "
              f"ALERTS: {n}   (avg {n / len(dates):.1f}/day)")
        if rr["reasons"]:
            print("flips rejected because of:")
            for reason, c in rr["reasons"].most_common():
                print(f"   {c:4d}  {reason}")
        if n == 0:
            print("no alerts in this period.")
            continue

        by_dir = alerts["direction"].value_counts().to_dict()
        print(f"by direction: {by_dir}")
        print("by day: " + ", ".join(f"{d}: {c}" for d, c in alerts["date"].value_counts().sort_index().items()))
        hours = Counter(alerts["time"].map(lambda t: t.hour))
        print("by hour (ET): " + ", ".join(f"{h:02d}h:{c}" for h, c in sorted(hours.items())))

        print("\nforward results, direction-adjusted (positive = moved the alert's way):")
        for h in horizons:
            print(f"  +{h:<2d} bars ({h * 5:>3d} min): {_fmt_stats(alerts[f'ret_{h}'])}")
        mid = horizons[len(horizons) // 2]
        print(f"  avg worst move against within {mid} bars: {alerts[f'mae_{mid}'].mean():.3f}%   "
              f"avg best move for: {alerts[f'mfe_{mid}'].mean():.3f}%")
        print("\nreference, ALL evaluated bars treated as long (market drift in this window):")
        for h in horizons:
            print(f"  +{h:<2d} bars: {_fmt_stats(result['baseline'][h])}")

        if show_alerts:
            cols = ["time", "direction", "close", "nearR", "nearR2", "nearS", "nearS2"] + [
                c for c in alerts.columns if c.startswith("ema_")]
            cols += list(config.TIMEFRAMES) + [f"ret_{h}" for h in horizons] + [f"mae_{mid}"]
            shown = alerts[[c for c in cols if c in alerts.columns]].copy()
            shown["time"] = shown["time"].map(lambda t: f"{t:%m-%d %H:%M}")
            print("\nalerts (time = start of the 5m bar that flipped; ret_* in %):")
            print(shown.round(3).to_string(index=False))


# --------------------------------------------------------------------------
# Data loading + CLI
# --------------------------------------------------------------------------
def load_bars(symbol: str, date: datetime.date, days: int, client) -> pd.DataFrame:
    """Alpaca 5m bars covering the last `days` trading days up to `date`
    plus config.LOOKBACK_DAYS of warm-up, with no Yahoo patch."""
    from alaDataManager import fetch_stock_data
    end = min(
        pd.Timestamp(datetime.combine(date, dt_time(20, 0))).tz_localize(config.EASTERN),
        pd.Timestamp.now(tz=config.EASTERN),
    )
    cal_days = math.ceil(days * 7 / 5) + 4          # generous: weekends / holidays
    return fetch_stock_data(
        symbol, config.BASE_RULE, lookback_days=cal_days + config.LOOKBACK_DAYS,
        client=client, end=end, patch_with_yahoo=False,
    )


def parse_window(text: str) -> tuple[dt_time, dt_time]:
    try:
        a, b = text.split("-")
        return dt_time.fromisoformat(a.strip()), dt_time.fromisoformat(b.strip())
    except ValueError:
        raise argparse.ArgumentTypeError("window must look like 09:30-15:30")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Replay alert rules over historical bars.")
    p.add_argument("symbols", nargs="*", help="Symbols, space and/or comma separated (default: config.SYMBOLS)")
    p.add_argument("--date", default=None, help="Last replay day, YYYY-MM-DD ET (default: today)")
    p.add_argument("--days", type=int, default=1, help="Number of trading days to replay (default 1)")
    p.add_argument("--compare", action="store_true",
                   help="Also run strict / EMA-only / raw-flip rule sets for comparison")
    p.add_argument("--window", type=parse_window, default=None, metavar="HH:MM-HH:MM",
                   help="Only evaluate bars whose start time is in this ET window, e.g. 09:30-15:30")
    p.add_argument("--delay-min", type=float, default=1.0,
                   help="Minutes after bar close that the live job would run (default 1)")
    p.add_argument("--horizons", default="6,12,24", help="Forward bars to measure (default 6,12,24)")
    p.add_argument("--csv", default=None, help="Save every alert to this CSV file")
    p.add_argument("--no-table", action="store_true", help="Don't print the per-alert table")
    args = p.parse_args(argv)

    syms = []
    for tok in args.symbols or []:
        syms += [s.strip().upper() for s in tok.split(",") if s.strip()]
    args.symbols = list(dict.fromkeys(syms)) or list(config.SYMBOLS)
    args.horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    args.date = (pd.Timestamp(args.date).date() if args.date
                 else datetime.now(config.EASTERN).date())
    return args


def main(argv=None, client=None, loader=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    analyzers = make_variants(args.compare)

    if loader is None:
        from alaDataManager import get_client
        client = client or get_client()
        loader = lambda sym: load_bars(sym, args.date, args.days, client)

    all_alerts = []
    for sym in args.symbols:
        try:
            df = loader(sym)
        except Exception:
            log.exception("Could not load bars for %s", sym)
            continue

        trading_days = sorted({ts.date() for ts in df.index if ts.date() <= args.date})
        dates = trading_days[-args.days:]
        if not dates:
            log.warning("No bars on or before %s for %s", args.date, sym)
            continue
        log.info("%s: replaying %s (%d bars loaded)", sym, ", ".join(map(str, dates)), len(df))

        result = replay_frame(df, sym, analyzers, set(dates), args.window,
                              args.delay_min, args.horizons)
        print_report(sym, result, dates, args.horizons, show_alerts=not args.no_table)
        for rr in result["variants"].values():
            all_alerts += rr["alerts"]

    if args.csv and all_alerts:
        pd.DataFrame(all_alerts).to_csv(args.csv, index=False)
        print(f"\nSaved {len(all_alerts)} alert rows to {args.csv}")

    if len(args.symbols) > 1 and all_alerts:
        summary = (pd.DataFrame(all_alerts).groupby(["variant", "symbol"]).size()
                   .rename("alerts").reset_index())
        print("\nALERTS PER SYMBOL\n" + summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
