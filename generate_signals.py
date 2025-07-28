import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

MULTIPLIER = 100

def generate_signal(iv_diff, threshold):
    if iv_diff > threshold:
        return 1  # Long straddle
    elif iv_diff < -threshold:
        return -1  # Short straddle
    else:
        return 0  # No trade

def load_data(ticker, feature_path, model_path):
    df = pd.read_csv(feature_path)
    model, features = joblib.load(model_path)
    return df, model, features

def apply_model(df, model, features, threshold):
    df = df.dropna(subset=features + ["atm_iv", "return", "straddle_cost"]).copy()
    X = df[features]
    df["predicted_vol"] = model.predict(X)
    df["iv_diff"] = df["predicted_vol"] - df["atm_iv"]
    df["trade_signal"] = df["iv_diff"].apply(lambda x: generate_signal(x, threshold))
    df["pnl"] = df["return"] * df["straddle_cost"]

    df["pnl_when_signal"] = np.where(
        df["trade_signal"] == 1,
        df["pnl"],
        np.where(df["trade_signal"] == -1, -df["pnl"], 0.0)
    )

    df["return_when_signal"] = df["return"] * df["trade_signal"]
    return df

def evaluate_performance(df, ticker, threshold):
    num_trades = (df["trade_signal"] != 0).sum()
    total_pnl = df["pnl_when_signal"].sum()
    avg_pnl = df["pnl_when_signal"].mean()
    win_rate = (df["pnl_when_signal"] > 0).sum() / max(num_trades, 1)
    long_trades = (df["trade_signal"] == 1).sum()
    short_trades = (df["trade_signal"] == -1).sum()

    print(f"\nSignal Performance for {ticker} (threshold={threshold}):")
    print(f" - Number of trades: {num_trades}")
    print(f" - Long straddles: {long_trades}")
    print(f" - Short straddles: {short_trades}")
    print(f" - Total PnL: ${total_pnl:.2f}")
    print(f" - Avg PnL per trade: ${avg_pnl:.2f}")
    print(f" - Win rate: {win_rate:.2%}")

def plot_vol_curves(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df["as_of"], df["predicted_vol"], label="Predicted Vol", marker="o")
    plt.plot(df["as_of"], df["realized_vol"], label="Realized Vol", marker="s")
    plt.plot(df["as_of"], df["atm_iv"], label="ATM IV", marker="^")
    plt.title(f"{ticker} Volatility Comparison Over Time")
    plt.xlabel("Date (as_of)")
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
    feature_path = f"{base_path}/features/{ticker}.csv"
    model_path = f"{model_dir}/elasticnet_model.pkl"
    signal_path = f"{base_path}/signals/{ticker}_signals.csv"

    df, model, features = load_data(ticker, feature_path, model_path)
    df = apply_model(df, model, features, threshold)
    evaluate_performance(df, ticker, threshold)

    if plot:
        plot_vol_curves(df, ticker)

    if save_output:
        os.makedirs(os.path.dirname(signal_path), exist_ok=True)
        df.to_csv(signal_path, index=False)
        print(f"\nSignals saved to: {signal_path}")

if __name__ == "__main__":
    backtest("AAPL", threshold=0.3)
