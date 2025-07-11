<a name="readme-top"></a>

# Trading Bot

Here’s a roadmap for taking you into a live, money-making bot—first in the simplest way possible, then in the absolute lowest-cost way.
1. The “Easiest” Path: Alpaca + Equity Market

    Open an Alpaca account

        Sign up at Alpaca Markets and get your paper-trading API keys.

        Once you’re happy with paper-trading results, upgrade to a funded (live) account.

    Install the Alpaca Python SDK

pip install alpaca-trade-api

Swap out yfinance for live data

from alpaca_trade_api import REST

API_KEY = "your_key"
API_SECRET = "your_secret"
BASE_URL = "https://paper-api.alpaca.markets"  # switch to live URL when going live
api = REST(API_KEY, API_SECRET, BASE_URL)

# Fetch barset for symbol
barset = api.get_barset("SPY", "day", start="2023-01-01", end="2025-07-07")
data = barset.df["SPY"]

Execute real orders
Replace your “if pred>price then buy” logic with:

    if signal == "buy":
        api.submit_order(
            symbol="SPY",
            qty=1,                # or cash_based=False, notional=1000
            side="buy",
            type="market",
            time_in_force="day"
        )
    elif signal == "sell":
        api.submit_order(
            symbol="SPY",
            qty=1,
            side="sell",
            type="market",
            time_in_force="day"
        )

    Deploy

        Local: schedule your script via cron (Linux/Mac) or Task Scheduler (Windows).

        Cloud: spin up an always-on instance on AWS Free Tier, Google Cloud (f1-micro), or Oracle Always Free VM.

    Go live

        Verify on paper for 2–4 weeks.

        Fund your account (start small!).

        Monitor performance, drawdowns, slippage.

Why this is easiest:

    Alpaca is zero-commission on US equities.

    Their SDK is well documented and Python-friendly.

    You keep your existing code structure—just swap data-fetch and add order calls.

2. The “Cheapest” Path: Crypto + Free Hosting

If you want to bypass minimums, day-trading rules, and keep costs at absolute zero:

    Pick a commission-free crypto exchange

        Binance (spot trading), Coinbase Pro (0.50% taker), or FTX US (free spot).

        All provide free REST/WebSocket APIs.

    Install a lightweight exchange client

pip install python-binance

Live data & order logic

    from binance.client import Client

    client = Client(api_key, api_secret)
    # Fetch klines
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2020")
    # Place market order
    client.order_market_buy(symbol="BTCUSDT", quantity=0.001)

    Use free compute

        Oracle Cloud Always Free: two small VMs.

        Raspberry Pi at home (one-time cost ~$35).

        Schedule via cron or a simple while True: socket.listen() loop.

    Run 24/7 with minimal capital

        Crypto markets never close—no “market-hours” restrictions.

        No Pattern-Day-Trading rule.

        Start with as little as $10.

Why this is cheapest:

    No monthly fees for your server/VPS.

    Crypto spot trading is commission-free or extremely low.

    No minimum account balances or equity thresholds.

3. Key Considerations Regardless of Path
Factor	Recommendation
Risk Management	Always cap position size (e.g. 1–5% of account equity).
Backtest Robustness	Include transaction costs, slippage, and realistic fills.
Regulatory	Equities: avoid PDT by using >$25 k or trade less than 4 day trades in 5 days; Crypto: largely unregulated but be mindful of KYC/AML.
Monitoring & Alerts	Use simple alerts (Slack/email/SMS) on order failures or drawdowns.
Logging	Persist logs of every trade, every signal, every error.
4. Summary Comparison
Path	Setup Effort	Fees	Markets	Drawbacks
Equities (Alpaca)	☑️☑️ (medium)	$0 commission	US stocks	Market hours, PDT rule
Crypto	☑️ (low)	$0–0.1%	24/7 crypto	Volatility, exchange reliability
Next Steps

    Decide market: equities vs crypto.

    Sign up: Alpaca or Binance.

    Adapt your code: swap data source & add order submission.

    Paper-trade: verify strategy.

    Fund & go live: start small, iterate.

Good luck—and remember that automated strategies come with real risk. Always start tiny, and scale only once you’ve proven consistent edge!
You said:
i'll do equities. so i guess i'll go with alpaca. i have a personal home server running ubuntu 24.04 LTS server version. 
ChatGPT said:

Here’s a step-by-step guide to get your Alpaca-powered bot up and running on your Ubuntu home server:
1. Prepare Your Server

    Update & install Python

sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv python3-pip git -y

Clone your repo

cd ~/projects
git clone https://github.com/yourusername/your-trading-bot.git
cd your-trading-bot

Create a virtual environment and install requirements

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt alpaca-trade-api

2. Secure Your Alpaca Keys

    Export as environment variables (add these to ~/.bashrc or /etc/environment)

export APCA_API_KEY_ID="your_key_id"
export APCA_API_SECRET_KEY="your_secret_key"
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"

Reload your shell

    source ~/.bashrc

3. Adapt Your Script for Live Trading

Below is skeleton code to replace your backtest loop with real Alpaca orders. You can drop this into a new file (e.g. live_bot.py) alongside your existing code:

import os
import pandas as pd
from alpaca_trade_api import REST
from your_module import prepare_features, train_model  # reuse your functions

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def fetch_historical(symbol: str, start: str) -> pd.DataFrame:
    barset = api.get_bars(symbol, "day", start=start, limit=1000).df
    # Alpaca’s df has columns: open, high, low, close, volume
    barset = barset.tz_convert(None).rename(columns={'close':'Close'})
    return barset

def live_trade(symbol: str, start: str):
    # 1) fetch & train
    data = fetch_historical(symbol, start)
    X, y = prepare_features(data)
    model = train_model(X, y)

    # 2) get today’s price
    today_bar = api.get_barset(symbol, "day", limit=1).df.iloc[-1]
    current_price = today_bar['close']

    # 3) predict tomorrow
    last_features = X.iloc[[-1]]
    pred_price = model.predict(last_features)[0]

    # 4) decide & submit
    position = api.get_position(symbol) if symbol in [p.symbol for p in api.list_positions()] else None
    if pred_price > current_price and (not position or int(position.qty) == 0):
        api.submit_order(symbol, qty=1, side='buy', type='market', time_in_force='day')
    elif pred_price <= current_price and position and int(position.qty) > 0:
        api.submit_order(symbol, qty=position.qty, side='sell', type='market', time_in_force='day')

    print(f"{symbol}: predicted {pred_price:.2f}, current {current_price:.2f}")

4. Automate Daily Runs

Your goal is to retrain on all data and place orders once per day shortly after market close (4 PM ET). In Denver (MDT), that’s 2 PM local.
A) Cron Job

    Open your crontab

crontab -e

Add this line (runs at 14:05 MDT Mon–Fri)

    5 14 * * 1-5   /home/kevin/projects/your-trading-bot/venv/bin/python \
      /home/kevin/projects/your-trading-bot/live_bot.py >> /home/kevin/logs/trading_bot.log 2>&1

    Tip: Ensure your server’s timezone is set to MDT:

    sudo timedatectl set-timezone America/Denver

B) (Optional) Systemd Timer

If you’d rather use systemd:

    Create a service /etc/systemd/system/trading_bot.service:

[Unit]
Description=Daily Alpaca Trading Bot

[Service]
WorkingDirectory=/home/kevin/projects/your-trading-bot
ExecStart=/home/kevin/projects/your-trading-bot/venv/bin/python live_bot.py
StandardOutput=append:/home/kevin/logs/trading_bot.log
StandardError=append:/home/kevin/logs/trading_bot.log

Create a timer /etc/systemd/system/trading_bot.timer:

[Unit]
Description=Run Trading Bot Daily

[Timer]
OnCalendar=Mon..Fri 14:05
Persistent=true

[Install]
WantedBy=timers.target

Enable & start

    sudo systemctl daemon-reload
    sudo systemctl enable --now trading_bot.timer

5. Monitoring & Logging

    Logs: check ~/logs/trading_bot.log daily.

    Email Alerts: if you want failures emailed, prefix your cron line with MAILTO=you@domain.com.

    Crash Recovery: with systemd, failures auto-restart if you add Restart=on-failure in the service block.

Final Checklist

Alpaca paper-trade successfully before going live

Environment variables stored securely

Timezone confirmed (timedatectl)

Cron or systemd timer running

    Logs rotating (e.g., via logrotate)

That’s it! You’re now running a live, daily-retraining bot on your own Linux server with zero hosting fees. Good luck—and trade responsibly!

<p align="right"><a href="#readme-top">back to top</a></p>
