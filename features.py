import numpy as np
import pandas as pd
import os
import sys
from arch import arch_model
import yfinance as yf
from pandas.tseries.offsets import BDay

TICKER = sys.argv[1].upper() 
MULTIPLIER = 100

def compute_garch_vol(ticker, as_of, window=500):
    df = yf.download(ticker, end=as_of, period=f"{window+5}d", progress=False, auto_adjust=False)
    df.dropna(inplace=True)
    df["ret"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    rets = df["ret"].replace([np.inf, -np.inf], np.nan).dropna()
    rets_pct = rets.iloc[-window:] * 100

    am = arch_model(rets_pct, vol="Garch", p=1, q=1, mean="Constant")
    res = am.fit(disp="off")

    f = res.forecast(horizon=1, reindex=False)
    var1_pct = f.variance.values[-1, 0]
    sigma1_pct = np.sqrt(var1_pct) 
    sigma1     = sigma1_pct / 100

    return sigma1 * np.sqrt(252)

def get_close_price(ticker, date, max_lookahead=5):
    for i in range(max_lookahead):
        target_date = date + BDay(i)
        df = yf.download(ticker, start=target_date, end=target_date + pd.Timedelta(days=1), progress=False, auto_adjust=False)
        if not df.empty:
            return df["Adj Close"].iloc[0].item()
    raise ValueError(f"[ERROR] No price data found for {ticker} within {max_lookahead} BDays after {date}")

def get_open_price(ticker, date):
    df = yf.download(ticker, start=date, end=date + pd.Timedelta(days=1), progress=False, auto_adjust=False)
    return df["Open"].iloc[0].item()

earn = pd.read_csv(
    f"data/earnings/{TICKER}.csv",
    parse_dates=["fiscalDateEnding", "reportedDate"]
)
earn = earn.sort_values("reportedDate").reset_index(drop=True)
earn["prev_quarter_EPS"] = earn["reportedEPS"].shift(1)
earn["estimatedEPS_change"] = earn["estimatedEPS"] - earn["prev_quarter_EPS"]
earn["estimatedEPS_pct_change"] = (
    earn["estimatedEPS_change"] / earn["prev_quarter_EPS"].replace(0, np.nan)
)
earn["as_of"] = earn["reportedDate"] - pd.Timedelta(days=1)
earn["garch_vol"] = earn["as_of"].apply(
    lambda dt: compute_garch_vol(TICKER, dt)
)

chain = pd.read_csv(
    f"data/option_chains/{TICKER}.csv",
    parse_dates=["date", "as_of"]
)
chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.date
iv_map = (
    chain[["as_of", "implied_volatility"]]
    .dropna()
    .drop_duplicates("as_of")
    .rename(columns={"implied_volatility": "atm_iv"})
)
earn = earn.merge(iv_map, on="as_of", how="left")
earn["iv_garch_spread"] = earn["atm_iv"] - earn["garch_vol"]
earn["surprise_norm"] = earn["surprise"] / earn["estimatedEPS"].abs().replace(0, np.nan)
earn["garch_vol_lag1"] = earn["garch_vol"].shift(1)
earn["atm_iv_lag1"] = earn["atm_iv"].shift(1)
earn["iv_garch_spread_lag1"] = earn["iv_garch_spread"].shift(1)

for col in ["garch_vol", "atm_iv", "iv_garch_spread"]:
    earn[f"{col}_roll5_mean"] = earn[col].rolling(window=5, min_periods=3).mean()
    earn[f"{col}_roll5_std"] = earn[col].rolling(window=5, min_periods=3).std()
    earn[f"{col}_roll10_mean"] = earn[col].rolling(window=10, min_periods=5).mean()
    earn[f"{col}_roll10_std"] = earn[col].rolling(window=10, min_periods=5).std()

labels = []
for idx, row in earn.iterrows():
    trade_date = row["reportedDate"]
    future_exp = chain[chain["expiration"] >= trade_date.date()]
    if future_exp.empty:
        labels.append({ ... })
        continue
    exp_date = future_exp["expiration"].min()
    day_chain = future_exp[future_exp["expiration"] == exp_date]
    if day_chain.empty:
        labels.append({k: np.nan for k in [
            "strike", "call_ask", "put_ask", "straddle_cost", "underlying",
            "underlying_t1", "abs_move", "return", "realized_vol", "breakeven_pct"
        ]})
        continue

    underlying_price = get_open_price(TICKER, trade_date)
    call = day_chain[day_chain["type"] == "call"].iloc[0]
    put = day_chain[day_chain["type"] == "put"].iloc[0]
    atm_strike = call["strike"]

    call_ask = call["ask"]
    put_ask = put["ask"]
    straddle_cost = (call_ask + put_ask) * MULTIPLIER

    underlying_t1 = get_close_price(TICKER, exp_date)
    abs_move = abs(underlying_t1 - atm_strike) * MULTIPLIER
    pnl = abs_move - straddle_cost
    ret = pnl / straddle_cost
    breakeven_pct = straddle_cost / underlying_price / MULTIPLIER

    log_return = np.log(underlying_t1 / underlying_price)
    realized_vol = abs(log_return) * np.sqrt(252)

    labels.append({
        "strike": atm_strike,
        "call_ask": call_ask,
        "put_ask": put_ask,
        "straddle_cost": straddle_cost,
        "underlying": underlying_price,
        "underlying_t1": underlying_t1,
        "abs_move": abs_move,
        "return": ret,
        "breakeven_pct": breakeven_pct,
        "realized_vol": realized_vol
    })


label_df = pd.DataFrame(labels)
features = pd.concat([earn.reset_index(drop=True), label_df], axis=1)
out_path = os.path.join("./data/features/", f"{TICKER}.csv")
features.to_csv(out_path, index=False)
print(features)
