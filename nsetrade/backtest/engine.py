"""A small, honest, long-only vectorised backtester.

Given an OHLCV frame and a strategy that produces a target position (0 or 1),
it simulates being in/out of the stock, applies a per-trade cost, and reports
the metrics that actually matter: CAGR, Sharpe, max drawdown, win-rate.

Signals are computed on bar *t* and acted on at bar *t+1*'s close to avoid
look-ahead bias. This is intentionally simple — it is a sanity check on an
edge, not a production execution simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..indicators import add_all

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Strategies: each returns a target-position Series (1 = long, 0 = flat)
# --------------------------------------------------------------------------


def strat_sma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long while fast SMA is above slow SMA (classic trend following)."""
    e = add_all(df)
    pos = (e[f"sma_{fast}"] > e[f"sma_{slow}"]).astype(float)
    return pos


def strat_rsi_ma(df: pd.DataFrame) -> pd.Series:
    """Buy oversold dips *within* an uptrend, exit when momentum overheats.

    Long when price is above the 200-SMA (uptrend filter) AND RSI < 35;
    stay long until RSI > 65. A pragmatic 'buy the dip in a bull' strategy.
    """
    e = add_all(df)
    above_trend = e["close"] > e["sma_200"]
    entry = above_trend & (e["rsi_14"] < 35)
    exit_ = e["rsi_14"] > 65

    pos = np.zeros(len(e))
    holding = False
    for i in range(len(e)):
        if holding:
            if bool(exit_.iloc[i]):
                holding = False
        elif bool(entry.iloc[i]):
            holding = True
        pos[i] = 1.0 if holding else 0.0
    return pd.Series(pos, index=e.index)


def strat_macd(df: pd.DataFrame) -> pd.Series:
    """Long while the MACD line is above its signal line."""
    e = add_all(df)
    return (e["macd"] > e["macd_signal"]).astype(float)


def strat_breakout(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Donchian breakout: enter on new N-day high, exit on N-day low."""
    hh = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    ll = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    pos = np.zeros(len(df))
    holding = False
    close = df["close"].values
    hhv, llv = hh.values, ll.values
    for i in range(len(df)):
        if np.isnan(hhv[i]):
            pos[i] = 0.0
            continue
        if holding:
            if close[i] < llv[i]:
                holding = False
        elif close[i] > hhv[i]:
            holding = True
        pos[i] = 1.0 if holding else 0.0
    return pd.Series(pos, index=df.index)


STRATEGIES = {
    "sma_crossover": strat_sma_crossover,
    "rsi_ma": strat_rsi_ma,
    "macd": strat_macd,
    "breakout": strat_breakout,
}


def list_strategies() -> list[str]:
    return list(STRATEGIES)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    metrics: dict
    equity_curve: pd.Series = field(repr=False)
    trades: int = 0

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.symbol} — {self.strategy}\n"
            f"  Period           : {m['start']} → {m['end']} "
            f"({m['bars']} bars)\n"
            f"  Strategy return  : {m['total_return']:+.1%}  "
            f"(CAGR {m['cagr']:+.1%})\n"
            f"  Buy & hold       : {m['buy_hold_return']:+.1%}\n"
            f"  Sharpe (annual)  : {m['sharpe']:.2f}\n"
            f"  Max drawdown     : {m['max_drawdown']:.1%}\n"
            f"  Trades           : {self.trades}  "
            f"(win-rate {m['win_rate']:.0%}, "
            f"exposure {m['exposure']:.0%})"
        )


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def backtest(
    symbol: str,
    df: pd.DataFrame,
    strategy: str = "rsi_ma",
    *,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Run ``strategy`` on ``df`` and compute performance metrics.

    Parameters
    ----------
    cost_bps:
        Round-trip-ish transaction cost in basis points applied on every
        position change (models brokerage + slippage). 5 bps = 0.05%.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}. Available: {', '.join(STRATEGIES)}"
        )
    if len(df) < 30:
        raise ValueError("need at least 30 bars to backtest")

    target = STRATEGIES[strategy](df).fillna(0.0).clip(0, 1)
    # Act on the next bar to avoid look-ahead: position held today was decided
    # by yesterday's signal.
    position = target.shift(1).fillna(0.0)

    daily_ret = df["close"].pct_change().fillna(0.0)
    gross = position * daily_ret

    # transaction cost whenever the position changes
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 10_000.0)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    buy_hold = (1.0 + daily_ret).cumprod()

    bars = len(df)
    years = bars / TRADING_DAYS
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0

    vol = net.std()
    sharpe = float(np.sqrt(TRADING_DAYS) * net.mean() / vol) if vol > 0 else 0.0

    # count trades + per-trade win rate
    entries = (position.diff() > 0)
    n_trades = int(entries.sum())
    win_rate = _trade_win_rate(position, daily_ret)

    metrics = {
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "bars": bars,
        "total_return": total_return,
        "cagr": cagr,
        "buy_hold_return": float(buy_hold.iloc[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "win_rate": win_rate,
        "exposure": float((position > 0).mean()),
    }
    return BacktestResult(
        symbol=symbol,
        strategy=strategy,
        metrics=metrics,
        equity_curve=equity,
        trades=n_trades,
    )


def _trade_win_rate(position: pd.Series, daily_ret: pd.Series) -> float:
    """Fraction of completed trades that were profitable."""
    pos = position.values
    ret = daily_ret.values
    wins = 0
    total = 0
    in_trade = False
    cum = 0.0
    for i in range(len(pos)):
        if pos[i] > 0 and not in_trade:
            in_trade = True
            cum = 0.0
        if in_trade:
            cum += ret[i]  # approx: sum of daily returns while held
        if in_trade and pos[i] == 0:
            in_trade = False
            total += 1
            if cum > 0:
                wins += 1
    if in_trade:  # close out open trade at end
        total += 1
        if cum > 0:
            wins += 1
    return wins / total if total else 0.0
