import os
import pandas as pd
import yfinance as yf
from requests.exceptions import HTTPError
from alpaca_trade_api import REST, TimeFrame
from your_module import prepare_features, train_model

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
DATA_FEED = os.getenv("APCA_API_DATA_FEED", "iex")

# The REST constructor no longer accepts a data_feed argument. Specify the feed
# when requesting bars instead. Only instantiate the client if credentials are
# available so the script can fall back to yfinance when running without them.
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2') if API_KEY and API_SECRET else None


def fetch_historical(symbol: str, start: str) -> pd.DataFrame:
    if api is None:
        data = yf.download(symbol, start=start, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.tz_localize(None)

    all_bars = []
    next_start = start
    try:
        while True:
            bars = api.get_bars(
                symbol, TimeFrame.Day, start=next_start, limit=1000, feed=DATA_FEED
            ).df
            if bars.empty:
                break
            # ``get_bars`` may return either timezone-aware or timezone-naive
            # indices depending on the API version. ``tz_localize(None)`` gracefully
            # handles both cases by removing the timezone when present without
            # raising an exception when it's already naive.
            bars = bars.tz_localize(None)
            bars.rename(columns={'close': 'Close'}, inplace=True)
            all_bars.append(bars)
            next_start = (bars.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            if len(bars) < 1000:
                break
        return pd.concat(all_bars)
    except HTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403):
            data = yf.download(symbol, start=start, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data.tz_localize(None)
        raise


def live_trade(symbol: str, start: str):
    data = fetch_historical(symbol, start)
    X, y = prepare_features(data)
    model = train_model(X, y)

    # Use the same data feed for the latest bar as for historical bars to avoid
    # HTTP 403 errors when the default feed isn't accessible. Fall back to
    # yfinance if the data API rejects the request.
    if api is not None:
        try:
            latest_bar = api.get_latest_bar(symbol, feed=DATA_FEED)
            current_price = latest_bar.c
        except HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                df = yf.download(symbol, period="1d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                current_price = df["Close"].iloc[-1]
            else:
                raise
    else:
        df = yf.download(symbol, period="1d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        current_price = df["Close"].iloc[-1]

    last_features = X.iloc[[-1]]
    pred_price = model.predict(last_features)[0]

    if api is not None:
        try:
            positions = {p.symbol: p for p in api.list_positions()}
            position = positions.get(symbol)
            if pred_price > current_price and (not position or int(position.qty) == 0):
                api.submit_order(symbol=symbol, qty=1, side='buy', type='market', time_in_force='day')
            elif pred_price <= current_price and position and int(position.qty) > 0:
                api.submit_order(symbol=symbol, qty=position.qty, side='sell', type='market', time_in_force='day')
        except HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                print("Trading not authorized; skipping order submission.")
            else:
                raise
    else:
        print("API credentials not provided; skipping order submission.")

    print(f"{symbol}: predicted {pred_price:.2f}, current {current_price:.2f}")


if __name__ == "__main__":
    live_trade("SPY", "2015-01-01")
