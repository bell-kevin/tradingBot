"""Gradient Boosting trading bot with simple hyperparameter tuning.

This script trains a ``GradientBoostingRegressor`` on historical data and
uses a validation split to search over leverage and dynamic leverage
factors. The best parameters on the validation set are then used for a
final backtest on unseen test data. The goal is to demonstrate a basic
approach to balancing performance with overfitting risk. This code is for
educational purposes only and comes with no guarantee of profitability.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from dataclasses import dataclass
from typing import List, Tuple


def fetch_data(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    return data


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def prepare_features(data: pd.DataFrame, window: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    df = data.copy()
    for i in range(1, window + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df = df.dropna()
    feature_cols = [f"lag_{i}" for i in range(1, window + 1)] + ["sma_5", "sma_10", "rsi_14"]
    X = df[feature_cols]
    y = df["Close"].shift(-1).dropna()
    X = X.iloc[:-1]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
    params = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }
    gbr = GradientBoostingRegressor(random_state=42)
    search = GridSearchCV(gbr, params, cv=TimeSeriesSplit(n_splits=5), n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_


@dataclass
class TradeResult:
    day: pd.Timestamp
    value: float


def simulate(
    prices: pd.Series,
    predictions: pd.Series,
    leverage: float,
    dynamic_factor: float,
    max_leverage: float = 10.0,
) -> List[TradeResult]:
    cash = 100.0
    position = 0.0
    debt = 0.0
    results: List[TradeResult] = []
    for day, pred in predictions.items():
        current_price = prices.loc[day]
        if pred > current_price and cash > 0.0:
            if dynamic_factor > 0:
                gain_pct = (pred - current_price) / current_price
                lev = 1 + dynamic_factor * gain_pct
                lev = max(1.0, min(lev, max_leverage))
            else:
                lev = leverage
            debt = (lev - 1) * cash
            position = lev * cash / current_price
            cash = 0.0
        elif pred <= current_price and position > 0.0:
            cash = position * current_price - debt
            position = 0.0
            debt = 0.0
        portfolio_value = cash + position * current_price - debt
        results.append(TradeResult(day=day, value=portfolio_value))
    return results


def evaluate_profit(results: List[TradeResult]) -> float:
    df = pd.DataFrame({"day": [r.day for r in results], "value": [r.value for r in results]}).set_index("day")
    start_value = 100.0
    final_value = df["value"].iloc[-1]
    profit = final_value - start_value
    trading_days = len(df)
    return profit / trading_days


def train_and_validate(
    symbol: str,
    start: str,
    window: int,
    leverage_values: List[float],
    dynamic_values: List[float],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[GradientBoostingRegressor, float, float]:
    data = fetch_data(symbol, start)
    X, y = prepare_features(data, window)
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * (train_ratio + val_ratio))
    train_X, train_y = X.iloc[:n_train], y.iloc[:n_train]
    val_X, val_y = X.iloc[n_train:n_val], y.iloc[n_train:n_val]
    model = train_model(train_X, train_y)
    best_score = -float("inf")
    best_leverage = leverage_values[0]
    best_dynamic = dynamic_values[0]
    preds = pd.Series(model.predict(val_X), index=val_X.index)
    for lev in leverage_values:
        for dyn in dynamic_values:
            results = simulate(val_y, preds, lev, dyn)
            score = evaluate_profit(results)
            if score > best_score:
                best_score = score
                best_leverage = lev
                best_dynamic = dyn
    print("Best leverage:", best_leverage)
    print("Best dynamic factor:", best_dynamic)
    print("Validation profit per day:", best_score)
    return model, best_leverage, best_dynamic


def backtest(
    model: GradientBoostingRegressor,
    data: pd.DataFrame,
    window: int,
    start_idx: int,
    leverage: float,
    dynamic_factor: float,
) -> List[TradeResult]:
    X, y = prepare_features(data, window)
    test_X = X.iloc[start_idx:]
    preds = pd.Series(model.predict(test_X), index=test_X.index)
    current_prices = data["Close"].loc[test_X.index]
    return simulate(current_prices, preds, leverage, dynamic_factor)


def summarize(results: List[TradeResult]) -> None:
    df = pd.DataFrame({"day": [r.day for r in results], "value": [r.value for r in results]}).set_index("day")
    daily_returns = df["value"].pct_change().fillna(0)
    start_value = 100.0
    final_value = df["value"].iloc[-1]
    profit = final_value - start_value
    trading_days = len(df)
    profit_per_day = profit / trading_days
    print("Number of results:", trading_days)
    print("Initial investment:", start_value)
    print("Final portfolio value:", final_value)
    print("Total profit:", profit)
    print(f"Duration: {trading_days} days")
    print(f"Average profit per day: {profit_per_day}")
    print("Yearly returns:")
    print(daily_returns.resample("YE").sum())


if __name__ == "__main__":
    symbol = "SPY"
    model, lev, dyn = train_and_validate(
        symbol,
        start="2015-01-01",
        window=10,
        leverage_values=[1.0, 1.5, 2.0, 3.0],
        dynamic_values=[0.0, 0.5, 1.0, 2.0],
    )
    data = fetch_data(symbol, "2015-01-01")
    X, _ = prepare_features(data, 10)
    start_idx = int(len(X) * 0.8)  # last 20% for testing
    results = backtest(model, data, 10, start_idx, lev, dyn)
    summarize(results)
