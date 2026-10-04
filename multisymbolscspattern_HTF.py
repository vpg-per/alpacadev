from __future__ import annotations
from alaDataManager import fetch_stock_data, drop_incomplete_last_candle
from dataAnalyzer import ServiceManager
from alertManager import AlertManager
import numpy as np
import os
import pandas as pd

objMgr = ServiceManager()
alertMgr = AlertManager()

def getTrendCrossvalue(symbol, interval) -> str :

    df = fetch_stock_data(symbol, interval)
    df = drop_incomplete_last_candle(df, interval)
    df = objMgr.calculate_rsi(df)
    last = df["rsitrend_cross"].iloc[-1]
    if last == 1:
        return "Bullish"
    elif last == -1:
        return "Bearish"
    
    return "none"


def process_alerts_request():
    bullish_symbols_data, bearish_symbols_data = [],[]
    interval = "30min"
    env_symbols = os.getenv("MULTI_STOCK_SYMBOLS", "")
    stocksymbols = [s.strip() for s in env_symbols.split(",") if s.strip()] or ['SPY']

    for symbol in stocksymbols:
        retval = getTrendCrossvalue(symbol, interval)
        if (retval == "Bullish"):
            bullish_symbols_data.append(symbol)
        elif(retval == "Bearish"):
            bearish_symbols_data.append(symbol)
    bullish_combinedmsg = ", ".join(bullish_symbols_data)
    bearish_combinedmsg = ", ".join(bearish_symbols_data)
    parts = []
    if bullish_symbols_data:
        parts.append(f"Bullish: {bullish_combinedmsg}")
    if bearish_symbols_data:
        parts.append(f"Bearish: {bearish_combinedmsg}")

    if parts:
        combinedmsg = f"Interval {interval}\n" + "\n".join(parts)
        print(combinedmsg)
        alertMgr.send_chart_alert(combinedmsg)
    else:
        print("Bias not changed for any input symbols.")


if __name__ == "__main__":
    process_alerts_request()