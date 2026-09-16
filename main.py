"""
Run with:  python main.py
"""

import sys
import argparse
from alaDataManager import fetch_stock_data
from dataAnalyzer import ServiceManager
from alertManager import AlertManager
from sectorperformance import generate_sector_chart
from chartDBManager import ChartDBManager
from datetime import datetime
import config

def define_input_symbols():
    parser = argparse.ArgumentParser(description="Process multiple stock symbols.")
    parser.add_argument("symbols", type=str, nargs="?", help="Comma-separated stock symbols")
    args = parser.parse_args()
    target_symbols = ["SPY"]
    if len(sys.argv) >= 2:
        target_symbols = [sym.strip().upper() for sym in args.symbols.split(",")]

    return target_symbols


def in_sector_chart_window(now: datetime) -> bool:
    """True when `now` (US/Eastern) falls between >6am and <5pm, on a
    minute strictly between 9 and 16 past the hour (i.e. minute 10-15).
    Intended to gate a periodically-invoked main.py so the sector chart
    only gets rebuilt roughly once an hour, during market-relevant hours.
    """
    return 6 < now.hour < 17 and 9 < now.minute < 16
    
def run_sector_chart():
    """Build the sector comparison chart and persist it as the single
    row in the Neon DB table (old row(s) removed first)."""
    png_path = generate_sector_chart()
    ChartDBManager().save_chart(png_path)
    print(f"Sector chart updated and saved to DB: {png_path}")


def main():
    symbols = define_input_symbols()

    objMgr = ServiceManager()
    alertMgr = AlertManager()
    for sym in symbols:
        row5m, row15m  = objMgr.analyze_stockdata(sym)

        if row5m is not None:
            row5m_index = row5m.name
            if isinstance(row5m_index, str):
                row5mdt = datetime.fromisoformat(row5m_index)
            elif isinstance(row5m_index, datetime):
                row5mdt = row5m_index
            else:
                # e.g. pandas.Timestamp or numpy.datetime64
                row5mdt = row5m_index.to_pydatetime()
            hour = row5mdt.hour
            minute = row5mdt.minute
            msg = f"{row5m['symbol']} 5m Bias changed({hour}:{minute:02d}) to {row5m['OverallBias']}, close is {row5m.close}, 15m bias is {row15m['OverallBias']} (prevbias: {row15m['PreviousBias']})"
            print(msg)
            alertMgr.send_chart_alert(msg)
        else:
            print(f"No bias change for {sym} stock on the latest bar.")

    now_et = datetime.now(config.EASTERN)
    if in_sector_chart_window(now_et):
        run_sector_chart()

    # combined = fetch_stock_data(symbol, "5min")
    # print(f"--- {symbol} ---")
    # print(combined)

    del objMgr, alertMgr


if __name__ == "__main__":
    main()