# ⚡ EdgeForge — Evidence-Based NSE Analysis & AI Trading Desk

> *Forge an edge from evidence.* (Python package: `nsetrade`.)

A Python library + command-line tool to analyse stocks listed on India's
**National Stock Exchange (NSE)**, detect high-probability technical patterns,
screen the whole market for opportunities, and backtest strategies before you
risk real money.

> ⚠️ **Disclaimer.** This software is for research and education only. It is
> **not** investment advice. Markets are risky; past performance does not
> guarantee future results. Always do your own due diligence and consider
> consulting a SEBI-registered advisor. You are solely responsible for your
> trades.

---

## What it does

| Area | What you get |
|------|--------------|
| **Data layer** | Pluggable providers, **all free**: a **yfinance** adapter (works out of the box) and a **Bhavcopy** adapter that serves the whole NSE from a local end-of-day store. |
| **Technical indicators** | RSI, MACD, SMA/EMA, Bollinger Bands, ATR, ADX, Stochastic, OBV, VWAP. |
| **Candlestick patterns** | Doji, Hammer, Shooting Star, Bullish/Bearish Engulfing, Morning/Evening Star, Piercing Line, Dark Cloud Cover. |
| **Chart patterns / signals** | Golden & Death cross, support/resistance, N‑day breakouts, 52‑week high/low proximity, RSI reversals, MACD crossovers. |
| **Structural patterns** | Cup & Handle, Darvas Box, Bull Flag, Double Bottom/Top, Ascending/Descending Triangle — heuristic detectors with breakout levels. |
| **Multi‑timeframe** | Run any analysis on **daily / weekly / monthly** candles (daily data is resampled, so it works for every provider). |
| **Watchlist** | Keep your own list of stocks (`watchlist.txt`, git‑ignored), import an NSE CSV (`EQUITY_L.csv` etc.), and scan just your picks. |
| **Fibonacci levels** | Auto‑computed retracement (23.6/38.2/50/61.8/78.6%) on the dominant swing and trend‑based extension targets — the industry‑standard ratios, drawn for you and fed to the AI. |
| **Opportunity ranker** | Scans a whole universe and ranks the most tradeable setups by **conviction + historical pattern edge + reward:risk** — surfaces the best evidence‑based candidates (probability, not a profit promise). |
| **AI analyst desk** | Optional: a panel of specialised AI agents (trend, pattern‑edge verifier, risk, devil's‑advocate) scrutinise a candidate and assign an **A–F grade** — a skeptic argument can veto a top grade. Raises shortlist *precision*, the honest version of "accuracy". |
| **AI trade thesis** | Optional: Claude reads the numeric analysis and writes a structured, grounded trade thesis (bias / setup / evidence / plan / risk), and can give a portfolio‑level read over the ranked opportunities. |
| **Pattern edge** | Backtest whether a chart pattern *historically worked* on a stock: hit‑rate, average forward return, target‑hit rate — evidence, not just a shape. |
| **Confluence** | Rank stocks where **daily + weekly + monthly** signals agree, with one weighted conviction score. |
| **Trade plan** | Auto entry / ATR stop / measured‑move target / position size from your capital and risk %. |
| **Alerts** | The live scanner pushes to console, a log file and optionally **Telegram**. |
| **Pro dashboard** | Interactive Plotly candlestick charts with pattern overlays, dark theme, multi‑tab layout. |
| **Screener** | Scan a universe (Nifty 50 / Nifty 500 / custom) and rank stocks by a combined bullish/bearish score with human‑readable reasons. |
| **Backtester** | Risk‑managed event‑driven backtest (ATR stop‑loss, take‑profit, position sizing) with win‑rate, CAGR, Sharpe, max drawdown, profit factor, expectancy and a full trade log. Plus **portfolio‑level** testing across a basket. |
| **Charts** | One‑command annotated PNG charts (price + MAs + Bollinger, volume, RSI, MACD) with detected patterns marked — glanceable on a phone. |
| **Live scanner** | Poll a watchlist on an interval and alert when a signal/pattern fires (free data is end-of-day/delayed). |
| **Dashboard** | A Streamlit web app over the whole toolkit: analyse, screen and backtest interactively. |

---

## Install

```bash
git clone <this repo>
cd trade
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # core (pandas, numpy, pyyaml)
pip install -e ".[yfinance]"     # add the free yfinance data provider
pip install -e ".[dashboard]"    # the dashboard (also pulls charts)
pip install -e ".[charts]"       # add PNG chart export (matplotlib)
pip install -e ".[dashboard]"    # add the Streamlit web dashboard
```

## Quick start

```bash
# Analyse a single stock and print indicators + detected patterns + signal
nsetrade analyse RELIANCE

# Screen the Nifty 50 and show the top bullish setups
nsetrade screen --universe nifty50 --top 15

# Risk-managed backtest (1% risk/trade, 2xATR stop, 4xATR target)
nsetrade backtest RELIANCE --strategy rsi_ma --years 3 --stop-atr 2 --target-atr 4

# Backtest the whole Nifty 50 as a portfolio
nsetrade backtest --universe nifty50 --strategy breakout --years 3

# Save an annotated PNG chart with patterns marked
nsetrade chart RELIANCE --out reliance.png

# Poll a basket on an interval and alert when signals fire
nsetrade watch --symbols RELIANCE,INFY,TCS --interval 300

# Launch the interactive web dashboard
streamlit run dashboard/app.py

# Build & scan your own watchlist
nsetrade watchlist add RELIANCE INFY TCS
nsetrade watchlist import EQUITY_L.csv          # import an NSE CSV export
nsetrade scan --watchlist --timeframes daily,weekly,monthly

# Find structural patterns (cup & handle, darvas, flag…) on weekly charts
nsetrade scan --universe nifty50 --timeframes weekly --breakouts-only

# Analyse / chart on a higher timeframe
nsetrade analyse RELIANCE --timeframe weekly
nsetrade chart RELIANCE --timeframe monthly

# Does a pattern historically work on this stock?
nsetrade edge RELIANCE --pattern cup_and_handle --forward 20

# Rank stocks where daily/weekly/monthly agree
nsetrade confluence --watchlist --aligned-only

# Auto trade plan (entry/stop/target/size)
nsetrade plan RELIANCE --capital 200000 --risk 0.01

# Rank the best tradeable setups across a universe (add --ai for a Claude read)
nsetrade opportunities --universe nifty50 --top 15
nsetrade opportunities --universe nifty50 --side short --ai

# Find stocks forming specific patterns (Cup & Handle, Darvas Box) + their edge
nsetrade picks --patterns cup_and_handle,darvas_box --universe nifty100

# Stocks breaking out to new highs over a chosen period (momentum screen)
nsetrade breakouts --period "52 weeks" --universe nifty100
nsetrade breakouts --period "6 months" --tol 2 --universe nifty100

# Cover the FULL NSE list, and precompute scans overnight for instant loading
nsetrade refresh-universe                       # download the ~2000-stock NSE list
nsetrade precompute --universe nse_all          # rank everything into a local cache
nsetrade opportunities --universe nse_all --cached   # read the cached ranking instantly

# Bulk EOD data for the whole NSE via official Bhavcopy (no per-stock API limits)
nsetrade fetch-bhavcopy --days 400              # build a local store (1 file/day)
nsetrade opportunities --universe nse_all --provider bhavcopy

# AI analyst desk — a panel of agents (trend, edge, risk, skeptic) grades setups A-F
nsetrade desk --universe nifty50 --top 3
nsetrade desk --symbols RELIANCE,TCS

# AI-written trade thesis (needs an Anthropic API key — see privacy note below)
nsetrade thesis RELIANCE --confluence

# List the patterns the toolkit knows about
nsetrade patterns
```

By default everything uses the free **yfinance** provider, so it works
immediately with no credentials.

### Watchlist & structural patterns

Build a personal watchlist and scan only your stocks across timeframes:

```bash
nsetrade watchlist add RELIANCE TATAMOTORS HDFCBANK   # add picks
nsetrade watchlist list                               # show them
nsetrade watchlist import ind_nifty500list.csv        # import an NSE CSV
nsetrade scan --watchlist                             # daily+weekly+monthly scan
```

The **scan** command runs heuristic detectors for Cup & Handle, Darvas Box,
Bull Flag, Double Bottom/Top and Triangles, and reports each hit's **breakout
level** and whether it is *forming* or already *breaking out*.

> ⚠️ These structural patterns are *visual* shapes with no rigorous definition,
> so detection is **approximate** — treat every hit as a candidate to confirm on
> the chart, not an automatic trade. The `chart` command (and dashboard) draw
> the levels so you can eyeball them.

## Full-NSE data via Bhavcopy (free, no rate limits)

For scanning the whole NSE reliably (yfinance rate-limits bulk requests), build
a local end-of-day store from NSE's official Bhavcopy — one file per trading
day instead of thousands of per-symbol calls:

1. `nsetrade fetch-bhavcopy --days 400` — downloads + stores history locally.
2. Use it with `--provider bhavcopy` (goes before the command), e.g.
   `nsetrade --provider bhavcopy opportunities --universe nse_all --top 20`,
   or pick **bhavcopy** in the dashboard's Data provider dropdown.

End-of-day only, and prices are raw (heuristic split adjustment is applied).

---

## Run it privately on your Windows laptop (no cloud, nothing exposed)

This toolkit is designed to run entirely on your own machine. Nothing is sent
anywhere except the data requests to your chosen provider (Yahoo or NSE). To
keep it private:

**One-time setup** — double-click `scripts\setup.bat` (or run it in a terminal).
It installs Python deps into a local `.venv` and creates your `config.yaml`.
You need [Python 3](https://www.python.org/downloads/) installed first (tick
*"Add Python to PATH"* in the installer).

**Run the dashboard** — double-click `scripts\run_dashboard.bat`. It starts the
server and opens **http://localhost:8501** in your browser automatically. Stop
it with `Ctrl+C`. No PowerShell typing needed.

**Launch it without finding the file each time** — double-click
`scripts\create_shortcut.bat` once to put a **"nsetrade Dashboard"** shortcut on
your Desktop. After that it's a single double-click. To open it automatically
**every time Windows starts**, press `Win+R`, type `shell:startup`, Enter, and
copy that Desktop shortcut into the folder that opens.

**Run CLI commands** — use `scripts\nsetrade.bat`, e.g.:
```bat
scripts\nsetrade.bat screen --universe nifty50 --top 15
scripts\nsetrade.bat chart RELIANCE --out reliance.png
```

### The AI thesis & your privacy

Everything in this toolkit runs **fully offline** except one opt-in feature: the
**AI trade thesis** (`nsetrade thesis` / the dashboard button). When you use it,
a **compact numeric summary** of the analysis (signal, indicators, detected
patterns + their historical edge, multi-timeframe verdicts) is sent to
Anthropic's API and Claude writes the thesis. It **never sends raw price data**,
and it only runs if you provide an Anthropic API key (`ai.api_key` in
`config.yaml` or `ANTHROPIC_API_KEY`). Leave the key empty and the platform is
100% local. Install with `pip install -e ".[ai]"`.

### Why this is private & secure

- **Localhost-only binding.** `.streamlit\config.toml` binds the dashboard to
  `127.0.0.1`, so it is reachable *only from this laptop* — not from your
  Wi-Fi, your office LAN, or the internet. No one else can open it.
- **Your secrets never leave the machine.** `config.yaml` (and `.env`,
  `*.token`) are in `.gitignore`, so anything you put there (e.g. your optional
  Anthropic API key) is **never committed or pushed** to GitHub. Keep the repo
  private too.
- **No telemetry.** Streamlit usage-stat reporting is turned off.
- **No external services.** There is no hosted backend, no account, no data
  sharing. Outbound traffic is only the price-data API calls you trigger.

> If you ever *want* to reach the dashboard from your phone on the same Wi-Fi,
> you'd change `address` to `0.0.0.0` in `.streamlit\config.toml` — but then add
> authentication first, and only do it on a trusted network. The default
> (localhost) needs no password precisely because nothing else can connect.

> ☝️ Treat your Anthropic API key like a password — keep it in `config.yaml`
> (git-ignored), and rotate it in the Anthropic console if it ever leaks.

---

## The "important patterns" — and how to use them

There is no holy grail, but these setups are the bread and butter of
technical traders. The toolkit detects all of them; the **signal engine**
combines them into a single score so you don't act on any one in isolation.

### Trend-following (works in trending markets)
- **Golden Cross** — 50‑day SMA crosses *above* the 200‑day SMA. Classic
  long‑term bullish signal. (Death Cross is the bearish mirror.)
- **MACD bullish crossover** — momentum turning up; strongest when it happens
  below the zero line and price is above the 200 SMA.
- **N‑day breakout** — price closes above its highest high of the last N days
  (e.g. 20 or 55, à la Turtle traders) **on rising volume**. Volume confirmation
  is what separates a real breakout from a fake‑out.

### Mean-reversion (works in range-bound markets)
- **RSI oversold/overbought** — RSI < 30 hints a bounce is due; RSI > 70 hints
  exhaustion. Best used *with* support/resistance, not alone.
- **Bollinger Band touches** — price tagging the lower band in an uptrend is a
  pullback‑buy candidate.

### Reversal candlesticks (timing entries/exits)
- **Bullish/Bearish Engulfing**, **Hammer / Shooting Star**,
  **Morning / Evening Star** — these mark potential turning points. They matter
  most *at* a support or resistance level, not in the middle of a range.

### Risk management beats any pattern
Every backtest here measures **max drawdown** for a reason. The edge from a
pattern is small; position sizing, stop‑losses (the toolkit reports ATR you can
size stops from) and cutting losers is what keeps you profitable. Treat signals
as a *shortlist to investigate*, never as automatic buy/sell orders.

---

## Project layout

```
nsetrade/
  data/        # pluggable data providers (yfinance, bhavcopy)
  indicators/  # pure-pandas technical indicators
  patterns/    # candlestick + chart pattern detectors
  signals/     # combine indicators+patterns into a scored signal
  screener/    # scan & rank a universe
  backtest/    # vectorised strategy backtester
  universe.py  # NSE symbol lists (nifty50, etc.)
  cli.py       # command-line interface
tests/         # unit tests (run with: pytest)
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests use synthetic data and never hit the network, so they run anywhere.
