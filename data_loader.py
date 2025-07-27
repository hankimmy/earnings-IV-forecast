import os
import json
import requests
import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
BASE_URL = "https://www.alphavantage.co/query"
TICKER = "AAPL"
OUT_DIR = "./data/option_chains"

def get_atm_strike(df: pd.DataFrame, as_of_date: str) -> float:
    lookup_ts = pd.to_datetime(as_of_date).normalize()
    price = df.loc[lookup_ts]['Adj Close']
    price = float(price.iloc[0])
    if price <= 25:
        strike_interval = 0.5
    elif price <= 200:
        strike_interval = 1.0
    else:
        strike_interval = 2.5
    return round(price / strike_interval) * strike_interval

def load_earnings_csv():
    resp = requests.get(
        BASE_URL,
        params={
            "function": "EARNINGS",
            "symbol":   TICKER,
            "apikey":   ALPHA_VANTAGE_API_KEY,
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    quarters = payload.get("quarterlyEarnings")
    earnings = pd.DataFrame(quarters)
    earnings.drop("reportTime", axis=1, inplace=True)
    earnings["reportedDate"] = pd.to_datetime(earnings["reportedDate"])
    out_path = os.path.join("./data/earnings", f"{TICKER}.csv")
    earnings.to_csv(out_path, index=False)

def load_options_chain_csv(earnings, stock_data):
    all_filtered = []
    for earn_date in earnings['reportedDate']:
        as_of = (pd.to_datetime(earn_date) - pd.tseries.offsets.BDay(1)).date().strftime('%Y-%m-%d')
        atm_strike = get_atm_strike(stock_data, as_of)
        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": TICKER,
            "date": as_of,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        payload = r.json()
        df = pd.DataFrame(payload["data"])

        df["strike"]      = pd.to_numeric(df["strike"], errors="coerce")
        df["expiration"]  = pd.to_datetime(df["expiration"]).dt.date
        df["as_of"]        = pd.to_datetime(df["date"]).dt.date

        earliest_exp = df["expiration"].min()
        df_atm_next = df[
            (df["strike"]     == atm_strike) &
            (df["expiration"] == earliest_exp)
        ]
        all_filtered.append(df_atm_next)

    result = pd.concat(all_filtered, ignore_index=True)
    out_path = os.path.join(OUT_DIR, f"{TICKER}.csv")
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} rows to {out_path}")

earnings_csv = os.path.join(".", "data", "earnings", f"{TICKER}.csv")
earnings = pd.read_csv(earnings_csv,parse_dates=["reportedDate"])

stock_data = yf.download(TICKER, start='2010-10-01', auto_adjust=False, progress=False)
stock_data = stock_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
stock_data.dropna(inplace=True)
stock_data.index = pd.to_datetime(stock_data.index)
stock_data.index = stock_data.index.normalize()
