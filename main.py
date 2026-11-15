"""
Run with:
    python main.py                          # watchlist from config.SYMBOLS, live
    python main.py SPY GL IWM               # explicit symbols (space and/or comma separated)
    python main.py SPY --tail 25            # also print last 25 candles per timeframe
    python main.py SPY --end "2026-11-13 11:45"      # debug: data as of 11/13 11:45 AM ET
    python main.py SPY --end "11/13/2026 11:45 AM"   # same, US format
    python main.py --dry-run                # analyze/print only: no alert, no state

--end is for debugging: times are US/Eastern, the analysis sees only bars
closed by then, and no alert is sent and no state is written.

Levels (levels.py) are attached to every alert, gated by the run time (ET):
  * before 09:30 -> previous-day levels only (pre-market levels don't exist yet)
  * from 10:00   -> opening range high/low (09:30-10:00) added
  * from 11:45   -> day high/low incl. pre-market (04:00-16:00 so far) added
Only the 2 nearest resistances and 2 nearest supports around the current
price are displayed (console log and Telegram message).

When several symbols alert in the SAME run, they are combined into ONE
Telegram message (one send_chart_alert call), one block per symbol.
"""
import argparse
import json
import logging
import sys

import pandas as pd

import config
from alertManager import AlertManager
from dataAnalyzer import AnalysisResult, SignalAnalyzer
from levels import SupportResistance, build_levels, nearest_line

log = logging.getLogger("main")
LABEL = {"Up": "Bullish", "Down": "Bearish"}


def parse_symbols(tokens) -> list[str]:
    """['SPY,GL', 'iwm', 'SPY'] -> ['SPY', 'GL', 'IWM'] (upper-cased, de-duplicated, order kept)."""
    symbols = []
    for token in tokens or []:
        for sym in token.split(","):
            sym = sym.strip().upper()
            if sym and sym not in symbols:
                symbols.append(sym)
    return symbols or list(config.SYMBOLS)


def parse_as_of(text: str) -> pd.Timestamp:
    """'2026-11-13 11:45' or '11/13/2026 11:45 AM' (US/Eastern unless an
    offset is given) -> tz-aware Eastern Timestamp. Future times are rejected."""
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError) as e:
        raise argparse.ArgumentTypeError(f"cannot parse date/time '{text}': {e}")
    ts = ts.tz_localize(config.EASTERN) if ts.tzinfo is None else ts.tz_convert(config.EASTERN)
    if ts > pd.Timestamp.now(tz=config.EASTERN):
        raise argparse.ArgumentTypeError(f"--end {ts:%Y-%m-%d %H:%M %Z} is in the future")
    return ts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MACD multi-timeframe alerts for one or more symbols.")
    parser.add_argument("symbols", nargs="*",
                        help="Symbols, space and/or comma separated (default: config.SYMBOLS)")
    parser.add_argument("--tail", type=int, default=5, metavar="N",
                        help="Print the last N candles per timeframe (e.g. 5 or 25; 0 = off). Default 5.")
    parser.add_argument("--tail-tf", default="", metavar="RULES",
                        help="Comma-separated timeframes to print, e.g. 5min,30min (default: all).")
    parser.add_argument("--end", type=parse_as_of, default=None, metavar="DATETIME",
                        help="Debug: analyze as of this US/Eastern date-time, e.g. \"2026-11-13 11:45\". "
                             "Implies --dry-run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze and print, but don't send alerts or record state.")
    args = parser.parse_args(argv)
    args.symbols = parse_symbols(args.symbols)
    args.tail_tf = [t.strip() for t in args.tail_tf.split(",") if t.strip()]
    args.dry_run = args.dry_run or args.end is not None
    return args


# ---- console output ---------------------------------------------------------
def print_tail(result: AnalysisResult, n: int, rules: list[str] | None = None) -> None:
    """print() the last n candles (with indicator columns) of each timeframe."""
    if n <= 0:
        return
    for rule, df in result.frames.items():
        if rules and rule not in rules:
            continue
        if df.empty:
            print(f"\n[{result.symbol} {rule}] no data")
            continue
        cols = [c for c in df.columns if c not in ("symbol", "interval")]
        print(f"\n[{result.symbol} {rule}] last {min(n, len(df))} candles")
        print(df.tail(n).to_string(columns=cols))


# ---- alert de-duplication (one alert per symbol/bar/direction) -------------
def load_state(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(path: str, state: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(state, f)
    except OSError as e:
        log.warning("Could not write state file %s: %s", path, e)


def signal_key(result: AnalysisResult) -> str:
    row = result.signal_row
    return f"{pd.Timestamp(row.name).isoformat()}|{row['MACDTrend']}"


def levels_for(result: AnalysisResult, now: pd.Timestamp) -> SupportResistance | None:
    """S/R levels for the session `now` falls in, built from the 5m frame.

    Time gates (levels.py): pre-market levels only from 09:30 ET, opening
    range (09:30-10:00) only from 10:00, day extremes incl. pre-market only
    from 11:45. Returns None (and logs) if levels can't be built -- the
    alert still goes out, just without the levels block.
    """
    base = result.frames.get(config.BASE_RULE)
    if base is None or base.empty:
        return None
    try:
        return build_levels(base, now.date(), now)
    except Exception as e:
        log.warning("Levels unavailable for %s: %s", result.symbol, e)
        return None


def build_message(result: AnalysisResult, sr: SupportResistance | None = None) -> str:
    row = result.signal_row
    ts = pd.Timestamp(row.name)
    base = next(iter(result.frames))
    parts = []
    if len(result.ema_cols) == 2:
        fast_col, slow_col = result.ema_cols
        fast, slow = row[fast_col], row[slow_col]
        parts.append(f"{fast_col.replace('ema_', 'EMA')} {fast} {'>' if fast > slow else '<'} "
                     f"{slow_col.replace('ema_', 'EMA')} {slow}")
    parts += [f"{rule} {r['MACDTrend']}" for rule, r in result.htf_rows.items()]
    msg = (
        f"{result.symbol} {base} MACD turned {LABEL[row['MACDTrend']]} ({ts:%H:%M}), "
        f"close is {row['close']}"
        #f"close is {row['close']}, {', '.join(parts)}"
    )
    if sr is not None:
        asof = f" ({pd.Timestamp(sr.asof):%H:%M})" if sr.asof is not None else ""
        msg += f"\nLevels{asof}: {nearest_line(sr, row['close'])}"
    return msg


def combine_messages(pending: list[tuple[str, str, str]]) -> str:
    """ONE Telegram message for every alert found in this run: a short
    header when there are several symbols, then one block per alert
    (blank line between them)."""
    body = "\n\n".join(m for _, _, m in pending)
    return f"{len(pending)} ALERTS\n\n{body}" if len(pending) > 1 else body


def default_analyzer() -> SignalAnalyzer:
    """Analyzer wired to Alpaca/Yahoo with ONE shared client for all symbols."""
    from alaDataManager import fetch_stock_data, get_client
    client = get_client()
    return SignalAnalyzer(
        fetcher=lambda sym, interval, end=None: fetch_stock_data(sym, interval, client=client, end=end)
    )


def main(argv=None, analyzer: SignalAnalyzer | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    log.info("Symbols: %s", ", ".join(args.symbols))
    if args.end is not None:
        log.info("DEBUG as-of %s -- dry run: no alerts, no state", f"{args.end:%Y-%m-%d %H:%M %Z}")

    analyzer = analyzer or default_analyzer()
    state = load_state(config.STATE_FILE)
    pending = []  # (symbol, key, message)
    failures = 0
    now = args.end or pd.Timestamp.now(tz=config.EASTERN)

    for sym in args.symbols:
        try:
            result = analyzer.analyze(sym, now=args.end)
        except Exception:
            failures += 1
            log.exception("Analysis failed for %s", sym)
            continue

        print_tail(result, args.tail, args.tail_tf)

        # Levels as of this run's time (time-gated, see levels.py). Only the
        # 2 nearest resistances and 2 nearest supports around the current
        # price are shown. Logged for every symbol, alert or not.
        sr = levels_for(result, now)
        if sr is not None:
            base = result.frames[config.BASE_RULE]
            last_close = float(base["close"].iloc[-1])
            log.info("%s (%.2f) Nearest R/S: %s", sym, last_close,
                     nearest_line(sr, last_close))

        if not result.has_signal:
            log.info("No alert for %s: %s", sym, result.reason)
            continue

        key = signal_key(result)
        if not args.dry_run and state.get(sym) == key:
            log.info("Already alerted for %s (%s)", sym, key)
            continue

        msg = build_message(result, sr)
        log.info("%s%s", "[dry-run, would alert] " if args.dry_run else "", msg)
        pending.append((sym, key, msg))

    # One Telegram message for ALL symbols that alerted in this run.
    if pending and not args.dry_run:
        try:
            AlertManager().send_chart_alert(combine_messages(pending))
        except Exception:
            failures += 1
            log.exception("Alert delivery failed; will retry on next run")
        else:
            for sym, key, _ in pending:
                state[sym] = key
            save_state(config.STATE_FILE, state)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
