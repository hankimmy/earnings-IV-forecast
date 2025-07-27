import numpy as np
import pandas as pd
from arch import arch_model
import yfinance as yf

TICKER = "AAPL"

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

earn = pd.read_csv(
    f"data/earnings/{TICKER}.csv",
    parse_dates=["fiscalDateEnding", "reportedDate"]
)
earn["as_of"] = earn["reportedDate"] - pd.Timedelta(days=1)
earn["garch_vol"] = earn["as_of"].apply(
    lambda dt: compute_garch_vol(TICKER, dt)
)

chain = pd.read_csv(
    f"data/option_chains/{TICKER}.csv",
    parse_dates=["date", "as_of"]
)
chain = chain.rename(columns={"implied_volatility": "atm_iv"})
tmp = earn.merge(chain[["as_of", "atm_iv"]], on="as_of", how="left")

features = tmp[[
    "as_of", "fiscalDateEnding", "estimatedEPS", "garch_vol", "atm_iv"
]].sort_values("as_of").reset_index(drop=True)

print(features)

def compute_next_day_return(ticker, as_of):
    df = yf.download(
        ticker,
        start=as_of.strftime("%Y-%m-%d"),
        end=(as_of + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False
    )
    closes = df["Adj Close"].dropna()
    if len(closes) < 2:
        return np.nan
    P0, P1 = closes.iloc[0], closes.iloc[1]
    return (P1 - P0) / P0

features['ret_t1'] = features['as_of'].apply(
    lambda dt: compute_next_day_return(TICKER, dt)
)

atm_chain = (
    chain[['as_of', 'mark', 'strike', 'expiration']]
    .drop_duplicates(subset=['as_of'])
)

features = features.merge(
    atm_chain.rename(columns={'mark':'straddle_cost', 'strike':'K'}),
    on='as_of', how='left'
)

def compute_straddle_payoff(row):
    df = yf.download(
        TICKER,
        start=row.expiration.strftime("%Y-%m-%d"),
        end=(row.expiration + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False
    )
    closes = df['Adj Close'].dropna()
    if closes.empty:
        return np.nan
    PT = closes.iloc[-1]
    return abs(PT - row.K)

features['straddle_payoff'] = features.apply(
    lambda row: compute_straddle_payoff(row), axis=1
)
features['straddle_pnl'] = features['straddle_payoff'] - features['straddle_cost']