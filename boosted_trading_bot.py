"""Gradient Boosting trading bot.

This example uses a GradientBoostingRegressor model trained on lagged closing prices
and runs a simple backtest. The strategy buys when tomorrow's predicted price is
higher than today's and sells otherwise. Results include average profit per day.

Disclaimer: this code is for research and educational purposes only and carries
no guarantee of profitability. Use at your own risk.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


@dataclass
class TradeResult:
    day: pd.Timestamp
    value: float


def fetch_data(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    return data


def prepare_features(data: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Create lagged OHLCV features and a few simple technical indicators."""
    df = data.copy()
    for i in range(1, window + 1):
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[f"{col.lower()}_lag_{i}"] = df[col].shift(i)
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["std_5"] = df["Close"].rolling(5).std()
    df = df.dropna()
    feature_cols = [
        f"{col.lower()}_lag_{i}"
        for i in range(1, window + 1)
        for col in ["Open", "High", "Low", "Close", "Volume"]
    ] + ["ma_5", "ma_10", "std_5"]
    X = df[feature_cols]
    y = df["Close"].shift(-1).dropna()
    X = X.iloc[:-1]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    params = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
    }
    gbr = GradientBoostingRegressor(random_state=42)
    search = GridSearchCV(gbr, params, cv=3, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_


def backtest(symbol: str, start: str, end: str | None = None) -> List[TradeResult]:
    data = fetch_data(symbol, start, end)
    feats, target = prepare_features(data)
    model = train_model(feats, target)

    results: List[TradeResult] = []
    cash = 10000.0
    position = 0.0
    threshold = 0.01

    for idx in range(len(feats)):
        # Keep feature names when predicting to avoid sklearn warnings
        X_row = feats.iloc[[idx]]
        pred_price = model.predict(X_row)[0]
        current_price = data["Close"].iloc[idx]
        if (pred_price - current_price) / current_price > threshold and cash > 0.0:
            position = cash / current_price
            cash = 0.0
        elif (current_price - pred_price) / current_price > threshold and position > 0.0:
            cash = position * current_price
            position = 0.0
        portfolio_value = cash + position * current_price
        results.append(TradeResult(day=feats.index[idx], value=portfolio_value))

    return results


def summarize(results: List[TradeResult]) -> None:
    df = pd.DataFrame([{"day": r.day, "value": r.value} for r in results]).set_index("day")
    daily_returns = df["value"].pct_change().fillna(0)
    start_value = 10000.0
    final_value = df["value"].iloc[-1]
    profit = final_value - start_value
    trading_days = len(df)
    profit_per_day = profit / trading_days

    print("Initial investment:", start_value)
    print("Final portfolio value:", final_value)
    print("Total profit:", profit)
    print(f"Duration: {trading_days} days")
    print(f"Average profit per day: {profit_per_day}")
    print("Daily returns:")
    print(daily_returns)
    print("Weekly returns:")
    print(daily_returns.resample("W").sum())
    print("Monthly returns:")
    print(daily_returns.resample("ME").sum())
    print("Yearly returns:")
    print(daily_returns.resample("YE").sum())


async def run_async(symbols: List[str], start: str, end: str | None = None) -> None:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, backtest, sym, start, end) for sym in symbols]
        all_results = await asyncio.gather(*tasks)
    for symbol, results in zip(symbols, all_results):
        print(f"\nResults for {symbol}")
        summarize(results)


if __name__ == "__main__":
    asyncio.run(run_async(["SPY"], start="2015-01-01"))
