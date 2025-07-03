"""Simple moving average crossover trading bot.

This script demonstrates a basic example of a trading strategy using a
moving average crossover. It downloads historical price data using the
`yfinance` library and generates buy/sell signals based on the
relationship between short and long moving averages.

**Disclaimer:** This code is provided for informational purposes only and
comes with no warranty of profitability or performance. Trading in
financial markets involves risk, and past performance does not guarantee
future results. Use this code at your own risk.
"""

import pandas as pd
import yfinance as yf


def download_data(symbol: str = "SPY", start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Download historical OHLC data for a symbol."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = yf.download(symbol, start=start, end=end, progress=False)
    # yfinance may return multi-index columns; drop the ticker level if present
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(1, axis=1)
    return data


def moving_average_crossover(data: pd.DataFrame, short_window: int = 50, long_window: int = 200) -> pd.DataFrame:
    """Calculate moving average crossover signals."""
    df = data.copy()
    df["short_ma"] = df["Close"].rolling(window=short_window).mean()
    df["long_ma"] = df["Close"].rolling(window=long_window).mean()
    df["signal"] = 0
    df.loc[df.index[short_window:], "signal"] = (
        df["short_ma"].iloc[short_window:] > df["long_ma"].iloc[short_window:]
    ).astype(int)
    df["positions"] = df["signal"].diff().fillna(0)
    return df


def run_backtest(symbol: str = "SPY", start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Run a simple backtest and print the cumulative return."""
    data = download_data(symbol=symbol, start=start, end=end)
    data = moving_average_crossover(data)
    returns = data["Close"].pct_change().fillna(0)
    strategy_returns = returns * data["signal"].shift(1).fillna(0)
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    print(f"Cumulative return for {symbol}: {total_return:%}")
    return cumulative_returns


if __name__ == "__main__":
    run_backtest()
