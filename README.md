# nsetrade — NSE Stock Pattern & Signal Toolkit

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
| **Data layer** | Pluggable providers. Ships with a free **yfinance** adapter (works out of the box) and a **Zerodha Kite** adapter for your paid API. |
| **Technical indicators** | RSI, MACD, SMA/EMA, Bollinger Bands, ATR, ADX, Stochastic, OBV, VWAP. |
| **Candlestick patterns** | Doji, Hammer, Shooting Star, Bullish/Bearish Engulfing, Morning/Evening Star, Piercing Line, Dark Cloud Cover. |
| **Chart patterns / signals** | Golden & Death cross, support/resistance, N‑day breakouts, 52‑week high/low proximity, RSI reversals, MACD crossovers. |
| **Screener** | Scan a universe (Nifty 50 / Nifty 500 / custom) and rank stocks by a combined bullish/bearish score with human‑readable reasons. |
| **Backtester** | Risk‑managed event‑driven backtest (ATR stop‑loss, take‑profit, position sizing) with win‑rate, CAGR, Sharpe, max drawdown, profit factor, expectancy and a full trade log. Plus **portfolio‑level** testing across a basket. |
| **Charts** | One‑command annotated PNG charts (price + MAs + Bollinger, volume, RSI, MACD) with detected patterns marked — glanceable on a phone. |
| **Live scanner** | Poll a watchlist on your Kite feed during market hours and alert when a signal/pattern fires. |
| **Dashboard** | A Streamlit web app over the whole toolkit: analyse, screen and backtest interactively. |

---

## Install

```bash
git clone <this repo>
cd trade
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # core (pandas, numpy, pyyaml)
pip install -e ".[yfinance]"     # add the free yfinance data provider
pip install -e ".[kite]"         # add the Zerodha Kite provider
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

# Live-watch a basket on your Kite feed (alerts when signals fire)
nsetrade watch --symbols RELIANCE,INFY,TCS --interval 300

# Launch the interactive web dashboard
streamlit run dashboard/app.py

# List the patterns the toolkit knows about
nsetrade patterns
```

By default everything uses the free **yfinance** provider, so it works
immediately with no credentials.

## Using your Zerodha Kite API

1. `pip install -e ".[kite]"`
2. Copy `config.example.yaml` to `config.yaml` and fill in your Kite
   `api_key` / `access_token` (see [Kite Connect docs](https://kite.trade/docs/connect/v3/)).
3. Run with `--provider kite`, e.g. `nsetrade analyse INFY --provider kite`.

The Kite access token expires daily; regenerate it through the Kite login flow
and update `config.yaml` (or set `KITE_ACCESS_TOKEN`).

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
  data/        # pluggable data providers (yfinance, kite)
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
