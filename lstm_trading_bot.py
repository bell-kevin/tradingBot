"""Simple LSTM-based trading bot.

This example trains a lightweight LSTM model on historical price data and
demonstrates concurrent backtesting using ``asyncio`` and ``ThreadPoolExecutor``.
The bot makes daily buy/sell decisions based on the model's price prediction
for the current day in an attempt to maximize profit.

Disclaimer: this code is for research and educational purposes only and comes
with no guarantee of profitability. Use at your own risk.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import torch
from torch import nn


def fetch_data(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch historical OHLC data with explicit auto adjustment."""
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    return data


def create_sequences(data: pd.Series, window: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs = []
    targets = []
    for i in range(len(data) - window):
        seqs.append(data.iloc[i:i+window].values)
        targets.append(data.iloc[i+window])
    X = torch.tensor(np.array(seqs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(-1)
    return X, y


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def train_model(X: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> LSTMModel:
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    return model


@dataclass
class TradeResult:
    day: pd.Timestamp
    value: float


def backtest(
    symbol: str,
    start: str,
    end: str | None = None,
    window: int = 5,
    train_split: float = 0.8,
    threshold: float = 0.0,
) -> List[TradeResult]:
    """Run a simple backtest using an LSTM model."""
    data = fetch_data(symbol, start, end)
    prices = data['Close']

    train_size = int(len(prices) * train_split)
    train_prices = prices.iloc[:train_size]
    scaler_min = train_prices.min()
    scaler_max = train_prices.max()
    scaled_train = (train_prices - scaler_min) / (scaler_max - scaler_min)
    X_train, y_train = create_sequences(scaled_train, window)
    model = train_model(X_train, y_train)

    scaled_prices = (prices - scaler_min) / (scaler_max - scaler_min)
    results: List[TradeResult] = []
    cash = 100.0
    position = 0.0

    for i in range(train_size, len(prices)):
        if i < window:
            continue
        seq = torch.tensor(
            scaled_prices.iloc[i - window : i].values, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(seq).item()
        pred_price = pred_scaled * (scaler_max - scaler_min) + scaler_min
        current_price = prices.iloc[i]

        if pred_price > current_price * (1 + threshold) and cash > 0.0:
            position = cash / current_price
            cash = 0.0
        elif pred_price < current_price * (1 - threshold) and position > 0.0:
            cash = position * current_price
            position = 0.0

        portfolio_value = cash + position * current_price
        results.append(TradeResult(day=prices.index[i], value=portfolio_value))

    return results


def summarize(results: List[TradeResult]) -> None:
    df = pd.DataFrame([
        {"day": r.day, "value": r.value} for r in results
    ]).set_index("day")
    daily_returns = df["value"].pct_change().fillna(0)

    # assume we start with $100 before any trades
    start_value = 100.0
    final_value = df["value"].iloc[-1]
    profit = final_value - start_value
    # number of trading days in the backtest
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
        print(f'\nResults for {symbol}')
        summarize(results)


if __name__ == '__main__':
    asyncio.run(run_async(['SPY'], start='2015-01-01'))
