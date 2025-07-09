import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neural_network import MLPRegressor


@dataclass
class TradeResult:
    day: pd.Timestamp
    value: float


def fetch_data(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    return data


def prepare_features(data: pd.DataFrame, window: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    df = data.copy()
    for i in range(1, window + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df = df.dropna()
    features = df[[f'lag_{i}' for i in range(1, window + 1)]]
    target = df['Close']
    return features, target


def train_model(features: pd.DataFrame, target: pd.Series) -> MLPRegressor:
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    model.fit(features, target)
    return model


def backtest(symbol: str, start: str, end: str | None = None) -> List[TradeResult]:
    data = fetch_data(symbol, start, end)
    feats, target = prepare_features(data)
    model = train_model(feats, target)

    results: List[TradeResult] = []
    position = 0.0
    cash = 100.0

    for idx in range(len(feats)):
        X = feats.iloc[[idx]].values
        pred_price = model.predict(X)[0]
        current_price = target.iloc[idx]
        if pred_price > current_price:
            position = cash / current_price
            cash = 0.0
        else:
            cash += position * current_price
            position = 0.0
        portfolio_value = cash + position * current_price
        results.append(TradeResult(day=feats.index[idx], value=portfolio_value))

    return results


def summarize(results: List[TradeResult]) -> None:
    df = pd.DataFrame([{'day': r.day, 'value': r.value} for r in results])
    df = df.set_index('day')
    daily = df['value'].pct_change().add(1).fillna(1)
    cumulative = daily.cumprod()

    print('Final portfolio value:', df['value'].iloc[-1])
    print('Daily return:', daily.iloc[-1] - 1)
    print('Weekly return:', daily.resample('W').prod().iloc[-1] - 1)
    print('Monthly return:', daily.resample('M').prod().iloc[-1] - 1)
    print('Yearly return:', daily.resample('Y').prod().iloc[-1] - 1)


async def run_async(symbols: List[str], start: str, end: str | None = None) -> None:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, backtest, sym, start, end) for sym in symbols]
        all_results = await asyncio.gather(*tasks)
    for symbol, results in zip(symbols, all_results):
        print(f'\nResults for {symbol}')
        summarize(results)


if __name__ == '__main__':
    asyncio.run(run_async(['SPY'], start='2015-01-01'))
