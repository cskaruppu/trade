"""Vectorised strategy backtester."""

from .engine import backtest, BacktestResult, STRATEGIES, list_strategies
from .risk import backtest_risk, RiskBacktestResult, Trade
from .portfolio import backtest_portfolio, PortfolioResult

__all__ = [
    "backtest",
    "BacktestResult",
    "STRATEGIES",
    "list_strategies",
    "backtest_risk",
    "RiskBacktestResult",
    "Trade",
    "backtest_portfolio",
    "PortfolioResult",
]
