<a name="readme-top"></a>

# 

## Example Trading Bot

This project includes a simple moving average crossover bot implemented in
`trading_bot.py`. It fetches historical price data and generates basic
buy/sell signals.

This repository also includes a neural network based bot in `neural_trading_bot.py` which trains an MLP model to predict prices. It runs data downloads and backtests concurrently for multiple symbols. For a more advanced example using an LSTM, see `lstm_trading_bot.py` which demonstrates parallel backtesting with a simple recurrent network.

### Getting Started on Windows 11 with VS Code

Follow these steps if you are new to Python and Git:

1. **Install Python** from [python.org](https://www.python.org/downloads/windows/) and select the option to *Add python.exe to PATH*.
2. **Install Git for Windows** from [git-scm.com](https://git-scm.com/download/win).
3. **Install Visual Studio Code** from [code.visualstudio.com](https://code.visualstudio.com/).
4. Open *Command Prompt* or *PowerShell* and clone this repository:
   ```bash
   git clone https://github.com/<your username>/tradingBot.git
   ```
5. Start VS Code and choose **File > Open Folder**, then select the `tradingBot` directory.
6. In VS Code open a terminal via **Terminal > New Terminal** and create a virtual environment:
   ```bash
   python -m venv .venv
   ```
7. Activate the environment:
   ```bash
   .venv\Scripts\activate
   ```
8. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
9. Run the example bot:
   ```bash
   python trading_bot.py
   ```
  Or try the neural version with `python neural_trading_bot.py`,
  or the LSTM version with `python lstm_trading_bot.py`.

### Usage

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run either of the scripts:
   ```bash
  python trading_bot.py
  python neural_trading_bot.py
  python lstm_trading_bot.py
  ```

### Disclaimer

This code is provided for informational purposes only and carries no
guarantee of profitability. Trading involves risk, and past performance
does not ensure future results. Use the bot at your own risk.

--------------------------------------------------------------------------------------------------------------------------
== We're Using GitHub Under Protest ==

This project is currently hosted on GitHub.  This is not ideal; GitHub is a
proprietary, trade-secret system that is not Free and Open Souce Software
(FOSS).  We are deeply concerned about using a proprietary system like GitHub
to develop our FOSS project. I have a [website](https://bellKevin.me) where the
project contributors are actively discussing how we can move away from GitHub
in the long term.  We urge you to read about the [Give up GitHub](https://GiveUpGitHub.org) campaign 
from [the Software Freedom Conservancy](https://sfconservancy.org) to understand some of the reasons why GitHub is not 
a good place to host FOSS projects.

If you are a contributor who personally has already quit using GitHub, please
email me at **bellKevin@pm.me** for how to send us contributions without
using GitHub directly.

Any use of this project's code by GitHub Copilot, past or present, is done
without our permission.  We do not consent to GitHub's use of this project's
code in Copilot.

![Logo of the GiveUpGitHub campaign](https://sfconservancy.org/img/GiveUpGitHub.png)

<p align="right"><a href="#readme-top">back to top</a></p>
