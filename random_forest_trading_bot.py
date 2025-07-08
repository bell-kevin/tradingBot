"""Random Forest trading bot.

This example trains a RandomForestRegressor on lagged OHLCV features and
performs a simple backtest. It buys when the next day's predicted close is
higher than today's close and sells otherwise.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
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
    df = data.copy()
    for i in range(1, window + 1):
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[f"{col.lower()}_lag_{i}"] = df[col].shift(i)
    df = df.dropna()
    feature_cols = [f"{col.lower()}_lag_{i}" for i in range(1, window + 1) for col in ["Open", "High", "Low", "Close", "Volume"]]
    X = df[feature_cols]
    y = df["Close"].shift(-1).dropna()
    X = X.iloc[:-1]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    params = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
    }
    rf = RandomForestRegressor(random_state=42)
    search = GridSearchCV(rf, params, cv=3, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_


def backtest(symbol: str, start: str, end: str | None = None) -> List[TradeResult]:
    data = fetch_data(symbol, start, end)
    feats, target = prepare_features(data)
    model = train_model(feats, target)

    results: List[TradeResult] = []
    cash = 10000.0
    position = 0.0

    for idx in range(len(feats)):
        X_row = feats.iloc[[idx]]  # preserve column names
        pred_price = model.predict(X_row)[0]
        current_price = data["Close"].iloc[idx]
        if pred_price > current_price and cash > 0.0:
            position = cash / current_price
            cash = 0.0
        elif pred_price <= current_price and position > 0.0:
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
