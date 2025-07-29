import pandas as pd
import joblib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def generate_signal(iv_diff, threshold):
    if iv_diff > threshold:
        return 1  # Long straddle
    elif iv_diff < -threshold:
        return -1  # Short straddle
    else:
        return 0  # No trade

def load_data(ticker, feature_path, model_path):
    df = pd.read_csv(feature_path, parse_dates=["reportedDate", "as_of"])
    model, features = joblib.load(model_path)
    return df, model, features

def apply_model(df, model, features, threshold):
    df = df.dropna(subset=features + ["atm_iv", "return", "straddle_cost"]).copy()
    X = df[features]
    df["predicted_vol"] = model.predict(X)
    df["iv_diff"]      = df["predicted_vol"] - df["atm_iv"]
    df["trade_signal"] = df["iv_diff"].apply(lambda x: generate_signal(x, threshold))
    df["pnl"] = df["return"] * df["straddle_cost"]
    df["pnl_when_signal"] = df["pnl"] * df["trade_signal"]
    df["return_when_signal"] = df["return"] * df["trade_signal"]
    return df

def evaluate_performance(df, label):
    num_trades   = (df["trade_signal"] != 0).sum()
    total_pnl    = df["pnl_when_signal"].sum()
    avg_pnl      = df.loc[df["trade_signal"] != 0, "pnl_when_signal"].mean()
    win_rate     = (df["pnl_when_signal"] > 0).sum() / max(num_trades, 1)
    longs        = (df["trade_signal"] == 1).sum()
    shorts       = (df["trade_signal"] == -1).sum()

    print(f"\n--- {label} signal performance ---")
    print(f"Trades:         {num_trades}")
    print(f"  Longs:        {longs}")
    print(f"  Shorts:       {shorts}")
    print(f"Total PnL:      ${total_pnl:,.2f}")
    print(f"Avg PnL/trade:  ${avg_pnl:,.2f}")
    print(f"Win rate:       {win_rate:.2%}")

def plot_vol_curves(df, ticker, label):
    plt.figure(figsize=(12,6))
    plt.plot(df["as_of"], df["predicted_vol"], label=f"{label} Predicted Vol", marker="o")
    plt.plot(df["as_of"], df["realized_vol"],  label=f"{label} Realized Vol",  marker="s")
    plt.plot(df["as_of"], df["atm_iv"],        label=f"{label} ATM IV",        marker="^")
    plt.title(f"{ticker} Vol Comparison: {label}")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def backtest(
    ticker,
    threshold=0.3,
    base_path="./data",
    model_dir="./models",
    save_output=True,
    plot=True
):
    feat_path   = f"{base_path}/features/{ticker}.csv"
    model_path  = f"{model_dir}/elasticnet_model_{ticker}.pkl"
    out_path    = f"{base_path}/signals/{ticker}_signals_oos.csv"

    df, model, features = load_data(ticker, feat_path, model_path)

    train = df[df["reportedDate"].dt.year <= 2022]
    test  = df[df["reportedDate"].dt.year > 2022]
    print(f"Data split: {len(train)} in-sample rows, {len(test)} out-of-sample rows")

    train_sig = apply_model(train, model, features, threshold)
    test_sig  = apply_model(test,  model, features, threshold)

    evaluate_performance(train_sig, "In-sample")
    evaluate_performance(test_sig,  "Out-of-sample")

    if plot:
        plot_vol_curves(test_sig,  ticker, "OOS")

    if save_output:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        test_sig.to_csv(out_path, index=False)
        print(f"\nOut-of-sample signals saved to: {out_path}")

def backtest_all(
    feature_dir     = "./data/features",
    model_dir       = "./models",
    threshold       = 0.5,
):
    oos_sigs = []
    for fname in os.listdir(feature_dir):
        if not fname.lower().endswith(".csv"):
            continue
        ticker = fname[:-4].upper()
        feat_path = os.path.join(feature_dir, fname)
        model_path = os.path.join(model_dir, f"elasticnet_model_{ticker}.pkl")

        df, model, features = load_data(ticker, feat_path, model_path)
        test = df[df["reportedDate"].dt.year > 2022]
        if test.empty:
            print(f"{ticker}: no out-of-sample data, skipping.")
            continue

        sig = apply_model(test, model, features, threshold)
        sig["ticker"] = ticker
        oos_sigs.append(sig[["as_of","ticker","pnl_when_signal"]])

    if not oos_sigs:
        print("No OOS signals generated.")
        return

    all_oos = pd.concat(oos_sigs, ignore_index=True)
    # port = all_oos.pivot_table(
    #     index="as_of",
    #     columns="ticker",
    #     values="pnl_when_signal",
    #     aggfunc="sum",
    # ).fillna(0)
    port_pnl = all_oos.pivot_table(
        index="as_of",
        columns="ticker",
        values="pnl_when_signal",
        aggfunc="sum",
    ).fillna(0)
    port_pnl["Pnl"] = port_pnl.sum(axis=1)

    daily_pnl = port_pnl["Pnl"]

    # performance metrics
    ann_factor = np.sqrt(252)
    sharpe = daily_pnl.mean() / daily_pnl.std() * ann_factor
    total_pnl = daily_pnl.sum()

    # cumulative PnL & drawdown
    cum_pnl = daily_pnl.cumsum()
    running_max = cum_pnl.cummax().clip(lower=0)
    drawdown = cum_pnl - running_max
    drawdown_pct = drawdown[running_max > 0] / running_max[running_max > 0] * 100
    max_dd_pct = drawdown_pct.min()

    print("\n=== Portfolio out-of-sample performance ===")
    print(f"  Total days traded: {len(daily_pnl)}")
    print(f"  Total PnL:         ${total_pnl:,.2f}")
    print(f"  Annualized Sharpe: {sharpe:.2f}")
    print(f"  Max drawdown (%):  {abs(max_dd_pct):.2f}%")
    # port["Pnl"] = port.sum(axis=1)

    # daily = port["Pnl"]
    # ann_factor = np.sqrt(252)
    # sharpe = daily.mean() / daily.std() * ann_factor

    # cum_pnl = daily.cumsum()
    # running_max = cum_pnl.cummax()
    # drawdown = cum_pnl - running_max

    # print("\n=== Portfolio out-of-sample performance ===")
    # print(f"  Total days traded: {len(daily)}")
    # print(f"  Annualized Sharpe: {sharpe:.2f}")


    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True,
                                   gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(cum_pnl.index, cum_pnl.values, label="Cumulative PnL")
    ax1.set_ylabel("Cum. PnL ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red")
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Date")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].upper() == "ALL":
        backtest_all()
    else:
        backtest(sys.argv[1].upper(), threshold=0.3)
