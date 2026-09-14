"""
Run with:  python main.py
"""

import sys
import argparse
from alaDataManager import fetch_stock_data
from dataAnalyzer import ServiceManager
from alertManager import AlertManager


def define_input_symbols():
    parser = argparse.ArgumentParser(description="Process multiple stock symbols.")
    parser.add_argument("symbols", type=str, nargs="?", help="Comma-separated stock symbols")
    args = parser.parse_args()
    target_symbols = ["SPY"]
    if len(sys.argv) >= 2:
        target_symbols = [sym.strip().upper() for sym in args.symbols.split(",")]

    return target_symbols

def main():
    symbols = define_input_symbols()

    objMgr = ServiceManager()
    alertMgr = AlertManager()
    for sym in symbols:
        row5m, row15m  = objMgr.analyze_stockdata(sym)

        if row5m is not None:
            msg = f"5m Bias just changed: {row5m['PreviousBias']} -> {row5m['OverallBias']} at {row5m.name}, 15m bias is {row15m['OverallBias']}"
            print(msg)
            alertMgr.send_chart_alert(msg)
        else:
            print("No bias change on the latest bar.")

    # combined = fetch_stock_data(symbol, "5min")
    # print(f"--- {symbol} ---")
    # print(combined)

    del objMgr, alertMgr


if __name__ == "__main__":
    main()