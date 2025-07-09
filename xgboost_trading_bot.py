"""XGBoost-based trading bot.

This script trains an ``XGBRegressor`` model on lagged closing prices
and performs a simple backtest.  It buys when the predicted price for
the next day is higher than today's close and sells otherwise.  The
results show the average profit per day.

This code is for educational purposes only and comes with no guarantee
of profitability.  Use at your own risk.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor


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


def prepare_features(data: pd.DataFrame, window: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    df = data.copy()
    for i in range(1, window + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df = df.dropna()
    X = df[[f"lag_{i}" for i in range(1, window + 1)]]
    y = df["Close"].shift(-1).dropna()
    X = X.iloc[:-1]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X, y)
    return model


def backtest(symbol: str, start: str, end: str | None = None) -> List[TradeResult]:
    data = fetch_data(symbol, start, end)
    feats, target = prepare_features(data)
    model = train_model(feats, target)

    results: List[TradeResult] = []
    cash = 100.0
    position = 0.0

    for idx in range(len(feats)):
        X_row = feats.iloc[[idx]]
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
    df = pd.DataFrame({"day": [r.day for r in results], "value": [r.value for r in results]}).set_index("day")
    daily_returns = df["value"].pct_change().fillna(0)
    start_value = 100.0
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
