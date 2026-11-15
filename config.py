"""All tunable settings in one place. Every module reads its *defaults* from
here, but every public function also accepts explicit overrides, so another
project can import the modules and ignore this file entirely."""

from datetime import time as dt_time
from zoneinfo import ZoneInfo

try:  # Alpaca is only needed by alaDataManager; keep config importable without it.
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    BASE_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)
except ImportError:  # pragma: no cover
    BASE_TIMEFRAME = None

# ---- Market / data ---------------------------------------------------------
SYMBOL = "SPY"                           # single-symbol default used by data helpers
SYMBOLS = ["SPY"]   
LOOKBACK_DAYS = 5
BASE_RULE = "5min"                       # pandas rule matching BASE_TIMEFRAME
EASTERN = ZoneInfo("America/New_York")   # zoneinfo: no .localize() pitfalls

# Feeds in priority order (earlier feeds win on overlapping timestamps).
# "sip" = regular + extended hours; "boats" = overnight. Add "iex" as a
# fallback if you lack "sip" access.
FEEDS = ["sip", "boats"]

# Alpaca columns dropped on ingest. Volume is KEPT (needed for VWAP).
ALPACA_DROP_COLUMNS = ["vwap", "trade_count"]

# Hours Yahoo covers. Used for the Yahoo patch gate and the Yahoo hour mask.
EXTENDED_START = dt_time(4, 0)
EXTENDED_END = dt_time(20, 0)
YAHOO_RECENT_MINUTES = 30

# (display label, pandas resample rule) pairs (kept for other modules)
RESAMPLE_RULES = [("15-Minute", "15min"), ("30-Minute", "30min"), ("1-Hour", "1h")]

# ---- Strategy --------------------------------------------------------------
# Higher timeframes built from the 5m base data. Each gets MACD (plus any
# indicators configured for it below). Add "1h" here if you want it back.
TIMEFRAMES = ["15min", "30min"]

# Timeframes whose MACD must AGREE with a 5m flip before an alert is sent.
# Use [] to disable confirmation.
CONFIRM_RULES = ["15min", "30min"]

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MIN_BARS = 60                            # 5m bars needed before trusting signals
MAX_SIGNAL_AGE_MIN = 10                  # ignore flips older than this

# 5m EMAs (columns ema_9 / ema_21 on the 5m frame). With EMA_FILTER on, a
# bullish flip also needs EMA_FAST > EMA_SLOW and a bearish flip needs
# EMA_FAST < EMA_SLOW on the flip bar.
EMA_FAST, EMA_SLOW = 9, 21
EMA_FILTER = True

# Confirmation-timeframe MACD trends that allow each 5m flip direction. Each
# timeframe in CONFIRM_RULES (15min and 30min) must be one of these values:
# Bullish (Up) needs Up or Neutral, Bearish (Down) needs Down or Neutral.
# Use ("Up",) / ("Down",) for the strict version.
HTF_CONFIRM = {
    "Up": ("Up", "Neutral"),
    "Down": ("Down", "Neutral"),
}

# Extra indicators, per timeframe. Key = timeframe rule ("5min", "15min", ...)
# or "*" for all timeframes; value = list of (name, params) from
# indicators.INDICATORS. Example:
#   {"5min": [("vwap", {})], "*": [("ema", {"periods": (50,)})]}
EXTRA_INDICATORS: dict = {}

# ---- Alert dedupe ----------------------------------------------------------
STATE_FILE = ".alert_state.json"
