"""Risk-aware, event-driven backtester.

Unlike the simple vectorised engine in :mod:`nsetrade.backtest.engine`, this one
simulates trade-by-trade so it can model the things that actually decide whether
a strategy makes money:

* **ATR-based stop-loss** — exit when price falls a multiple of ATR below entry.
* **Take-profit target** — optional ATR-multiple profit target.
* **Position sizing** — risk a fixed fraction of equity per trade, sized from the
  stop distance (the textbook "risk 1% per trade" rule).
* A full **trade log** plus expectancy / profit-factor / payoff statistics.

Entries are taken on the bar *after* the signal fires (next-day open) to avoid
look-ahead bias. Stops/targets are checked intrabar against high/low.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..indicators import add_all, atr as atr_indicator
from .engine import STRATEGIES, TRADING_DAYS, _max_drawdown


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: float
    reason: str  # "stop" | "target" | "signal" | "eod"

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        return self.exit_price / self.entry_price - 1.0


@dataclass
class RiskBacktestResult:
    symbol: str
    strategy: str
    metrics: dict
    equity_curve: pd.Series = field(repr=False)
    trades: list[Trade] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.symbol} — {self.strategy} (risk-managed)\n"
            f"  Period           : {m['start']} → {m['end']} ({m['bars']} bars)\n"
            f"  Final equity     : ₹{m['final_equity']:,.0f} "
            f"(start ₹{m['start_equity']:,.0f})\n"
            f"  Total return     : {m['total_return']:+.1%}  "
            f"(CAGR {m['cagr']:+.1%})\n"
            f"  Buy & hold       : {m['buy_hold_return']:+.1%}\n"
            f"  Sharpe (annual)  : {m['sharpe']:.2f}\n"
            f"  Max drawdown     : {m['max_drawdown']:.1%}\n"
            f"  Trades           : {m['num_trades']}  "
            f"(win-rate {m['win_rate']:.0%})\n"
            f"  Avg win / loss   : {m['avg_win']:+.1%} / {m['avg_loss']:+.1%}  "
            f"(payoff {m['payoff']:.2f})\n"
            f"  Profit factor    : {m['profit_factor']:.2f}  "
            f"expectancy {m['expectancy']:+.2%}/trade\n"
            f"  Exit breakdown   : {m['exit_counts']}"
        )


def backtest_risk(
    symbol: str,
    df: pd.DataFrame,
    strategy: str = "rsi_ma",
    *,
    capital: float = 100_000.0,
    risk_per_trade: float = 0.01,
    stop_atr: float = 2.0,
    target_atr: float | None = 4.0,
    atr_period: int = 14,
    cost_bps: float = 5.0,
    max_position_frac: float = 1.0,
) -> RiskBacktestResult:
    """Event-driven backtest with stops, targets and position sizing.

    Parameters
    ----------
    capital:           starting equity (₹).
    risk_per_trade:    fraction of equity risked per trade (0.01 = 1%).
    stop_atr:          stop distance in ATR multiples below entry.
    target_atr:        take-profit distance in ATR multiples (None = no target).
    cost_bps:          per-side transaction cost in basis points.
    max_position_frac: cap on fraction of equity deployed in one position.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}. Available: {', '.join(STRATEGIES)}"
        )
    if len(df) < atr_period + 5:
        raise ValueError("not enough bars to backtest")

    target_pos = STRATEGIES[strategy](df).fillna(0.0).clip(0, 1)
    atr = atr_indicator(add_all(df), atr_period)

    dates = df.index
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    want_long = target_pos.values
    atr_v = atr.values
    cost = cost_bps / 10_000.0

    equity = float(capital)
    start_equity = equity
    in_trade = False
    shares = 0.0
    entry_price = stop = 0.0
    target = None
    entry_date = None
    trades: list[Trade] = []
    equity_series = np.empty(len(df))

    for i in range(len(df)):
        # mark-to-market equity for the curve (realised cash + unrealised pnl)
        if in_trade:
            equity_series[i] = equity + shares * (close[i] - entry_price)
        else:
            equity_series[i] = equity

        if in_trade:
            exit_price = None
            reason = None
            # intrabar: assume the stop can trigger before the target (conservative)
            if low[i] <= stop:
                exit_price, reason = stop, "stop"
            elif target is not None and high[i] >= target:
                exit_price, reason = target, "target"
            elif want_long[i] == 0:
                exit_price, reason = close[i], "signal"
            elif i == len(df) - 1:
                exit_price, reason = close[i], "eod"

            if exit_price is not None:
                gross = shares * (exit_price - entry_price)
                fees = cost * shares * (entry_price + exit_price)
                equity = equity + gross - fees
                trades.append(
                    Trade(entry_date, entry_price, dates[i], exit_price,
                          shares, reason)
                )
                in_trade = False
                shares = 0.0
                equity_series[i] = equity
                continue

        # enter on a fresh 0->1 transition (signal fired on the previous bar)
        prev2 = want_long[i - 2] if i >= 2 else 0.0
        if not in_trade and i >= 1 and want_long[i - 1] == 1 and prev2 == 0 \
                and not np.isnan(atr_v[i]) and atr_v[i] > 0:
            entry_price = open_[i]
            stop = entry_price - stop_atr * atr_v[i]
            target = entry_price + target_atr * atr_v[i] if target_atr else None
            risk_amount = equity * risk_per_trade
            per_share_risk = max(entry_price - stop, 1e-9)
            size = risk_amount / per_share_risk
            max_shares = (equity * max_position_frac) / entry_price
            shares = float(min(size, max_shares))
            if shares > 0:
                entry_date = dates[i]
                in_trade = True
                equity_series[i] = equity  # cash unchanged at the entry instant

    equity_curve = pd.Series(equity_series, index=dates)
    return _summarise(symbol, strategy, df, equity_curve, trades, start_equity)


def _summarise(symbol, strategy, df, equity_curve, trades, start_equity):
    daily_ret = equity_curve.pct_change().fillna(0.0)
    bars = len(df)
    years = bars / TRADING_DAYS
    final_equity = float(equity_curve.iloc[-1])
    total_return = final_equity / start_equity - 1.0
    cagr = (final_equity / start_equity) ** (1 / years) - 1.0 if years > 0 else 0.0
    vol = daily_ret.std()
    sharpe = float(np.sqrt(TRADING_DAYS) * daily_ret.mean() / vol) if vol > 0 else 0.0

    rets = [t.return_pct for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in trades if t.pnl <= 0)

    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.reason] = exit_counts.get(t.reason, 0) + 1

    metrics = {
        "start": df.index[0].date().isoformat(),
        "end": df.index[-1].date().isoformat(),
        "bars": bars,
        "start_equity": start_equity,
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "buy_hold_return": float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity_curve),
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "payoff": (np.mean(wins) / -np.mean(losses))
        if wins and losses and np.mean(losses) != 0 else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        if gross_win > 0 else 0.0,
        "expectancy": float(np.mean(rets)) if rets else 0.0,
        "exit_counts": exit_counts,
    }
    return RiskBacktestResult(symbol, strategy, metrics, equity_curve, trades)
