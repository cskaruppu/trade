"""Portfolio-level backtest across many stocks.

Runs the risk-managed engine on each symbol with an equal slice of capital, then
aggregates the per-symbol equity curves into one portfolio curve. This is a
simplified model: capital is allocated equally and statically (no rebalancing,
no shared cash pool), which is enough to see whether an edge survives across a
basket rather than on a single lucky stock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import TRADING_DAYS, _max_drawdown
from .risk import RiskBacktestResult, backtest_risk


@dataclass
class PortfolioResult:
    strategy: str
    metrics: dict
    equity_curve: pd.Series = field(repr=False)
    per_symbol: dict[str, RiskBacktestResult] = field(default_factory=dict, repr=False)
    errors: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Portfolio — {self.strategy} ({m['num_symbols']} symbols)",
            f"  Period           : {m['start']} → {m['end']}",
            f"  Total return     : {m['total_return']:+.1%}  "
            f"(CAGR {m['cagr']:+.1%})",
            f"  Sharpe (annual)  : {m['sharpe']:.2f}",
            f"  Max drawdown     : {m['max_drawdown']:.1%}",
            f"  Total trades     : {m['total_trades']}  "
            f"(win-rate {m['win_rate']:.0%})",
        ]
        # show contributors ranked by return; split best/worst only when the
        # basket is large enough that the two lists wouldn't overlap.
        ranked = sorted(
            self.per_symbol.items(),
            key=lambda kv: kv[1].metrics["total_return"],
            reverse=True,
        )
        if ranked and len(ranked) <= 6:
            lines.append("  Contributors     :")
            for sym, r in ranked:
                lines.append(f"    {sym:<12} {r.metrics['total_return']:+.1%}")
        elif ranked:
            lines.append("  Top contributors :")
            for sym, r in ranked[:3]:
                lines.append(f"    {sym:<12} {r.metrics['total_return']:+.1%}")
            lines.append("  Worst contributors:")
            for sym, r in ranked[-3:]:
                lines.append(f"    {sym:<12} {r.metrics['total_return']:+.1%}")
        if self.errors:
            lines.append(f"  Skipped          : {len(self.errors)} symbols")
        return "\n".join(lines)


def backtest_portfolio(
    frames: dict[str, pd.DataFrame],
    strategy: str = "rsi_ma",
    *,
    capital: float = 1_000_000.0,
    **risk_kwargs,
) -> PortfolioResult:
    """Backtest ``strategy`` across ``frames`` (symbol -> OHLCV) and aggregate.

    Capital is split equally across symbols. Extra keyword args are forwarded to
    :func:`nsetrade.backtest.risk.backtest_risk` (stop_atr, risk_per_trade, ...).
    """
    if not frames:
        raise ValueError("no symbols provided")

    per_symbol: dict[str, RiskBacktestResult] = {}
    errors: dict[str, str] = {}
    slice_capital = capital / len(frames)

    curves: list[pd.Series] = []
    for sym, df in frames.items():
        try:
            res = backtest_risk(sym, df, strategy=strategy,
                                capital=slice_capital, **risk_kwargs)
        except Exception as exc:  # noqa: BLE001 - skip bad symbols, keep going
            errors[sym] = str(exc)
            continue
        per_symbol[sym] = res
        curves.append(res.equity_curve.rename(sym))

    if not curves:
        raise ValueError(f"no symbols could be backtested; errors: {errors}")

    # align on the union of dates; forward-fill gaps, seed missing starts at the
    # symbol's slice capital so a late-starting stock doesn't distort the curve.
    matrix = pd.concat(curves, axis=1).sort_index()
    matrix = matrix.ffill().fillna(slice_capital)
    portfolio = matrix.sum(axis=1)

    daily_ret = portfolio.pct_change().fillna(0.0)
    bars = len(portfolio)
    years = bars / TRADING_DAYS
    total_return = float(portfolio.iloc[-1] / capital - 1.0)
    cagr = (portfolio.iloc[-1] / capital) ** (1 / years) - 1.0 if years > 0 else 0.0
    vol = daily_ret.std()
    sharpe = float(np.sqrt(TRADING_DAYS) * daily_ret.mean() / vol) if vol > 0 else 0.0

    all_trades = [t for r in per_symbol.values() for t in r.trades]
    wins = sum(1 for t in all_trades if t.pnl > 0)

    metrics = {
        "num_symbols": len(per_symbol),
        "start": portfolio.index[0].date().isoformat(),
        "end": portfolio.index[-1].date().isoformat(),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(portfolio),
        "total_trades": len(all_trades),
        "win_rate": wins / len(all_trades) if all_trades else 0.0,
    }
    return PortfolioResult(strategy, metrics, portfolio, per_symbol, errors)
