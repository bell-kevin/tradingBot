import os
import pandas as pd
from alpaca_trade_api import REST
from your_module import prepare_features, train_model

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')


def fetch_historical(symbol: str, start: str) -> pd.DataFrame:
    all_bars = []
    next_start = start
    while True:
        bars = api.get_bars(symbol, "day", start=next_start, limit=1000).df
        if bars.empty:
            break
        bars = bars.tz_convert(None)
        bars.rename(columns={'close': 'Close'}, inplace=True)
        all_bars.append(bars)
        next_start = (bars.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if len(bars) < 1000:
            break
    return pd.concat(all_bars)


def live_trade(symbol: str, start: str):
    data = fetch_historical(symbol, start)
    X, y = prepare_features(data)
    model = train_model(X, y)

    today_bar = api.get_barset(symbol, "day", limit=1).df.iloc[-1]
    current_price = today_bar['close']

    last_features = X.iloc[[-1]]
    pred_price = model.predict(last_features)[0]

    positions = {p.symbol: p for p in api.list_positions()}
    position = positions.get(symbol)
    if pred_price > current_price and (not position or int(position.qty) == 0):
        api.submit_order(symbol=symbol, qty=1, side='buy', type='market', time_in_force='day')
    elif pred_price <= current_price and position and int(position.qty) > 0:
        api.submit_order(symbol=symbol, qty=position.qty, side='sell', type='market', time_in_force='day')

    print(f"{symbol}: predicted {pred_price:.2f}, current {current_price:.2f}")


if __name__ == "__main__":
    live_trade("SPY", "2015-01-01")
