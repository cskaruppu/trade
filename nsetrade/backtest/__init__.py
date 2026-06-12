"""Vectorised strategy backtester."""

from .engine import backtest, BacktestResult, STRATEGIES, list_strategies

__all__ = ["backtest", "BacktestResult", "STRATEGIES", "list_strategies"]
