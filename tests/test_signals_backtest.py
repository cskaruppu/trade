"""Tests for the signal engine, screener and backtester."""

import pandas as pd
import pytest

from nsetrade.backtest import backtest, list_strategies
from nsetrade.data.base import DataProvider
from nsetrade.screener import screen
from nsetrade.signals.engine import signal_for_frame


def test_signal_for_frame(trending_df):
    sig = signal_for_frame("TEST", trending_df)
    assert sig.symbol == "TEST"
    assert sig.verdict in {"Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"}
    assert isinstance(sig.reasons, list)
    row = sig.as_row()
    assert set(["symbol", "close", "score", "verdict"]).issubset(row)


@pytest.mark.parametrize("strategy", list_strategies())
def test_backtest_runs_all_strategies(trending_df, strategy):
    res = backtest("TEST", trending_df, strategy=strategy)
    m = res.metrics
    assert -1.0 <= m["max_drawdown"] <= 0.0
    assert 0.0 <= m["exposure"] <= 1.0
    assert 0.0 <= m["win_rate"] <= 1.0
    assert res.equity_curve.iloc[-1] > 0
    assert "Sharpe" in res.summary()


def test_backtest_unknown_strategy(trending_df):
    with pytest.raises(ValueError):
        backtest("TEST", trending_df, strategy="nope")


def test_backtest_no_lookahead_flat_when_signal_never_fires(make_ohlcv):
    import numpy as np
    # strictly declining price: rsi_ma (needs uptrend) should stay flat
    prices = np.linspace(200, 100, 260)
    df = make_ohlcv(prices)
    res = backtest("DOWN", df, strategy="rsi_ma")
    assert res.metrics["exposure"] == 0.0
    assert res.trades == 0


class _FakeProvider(DataProvider):
    name = "fake"

    def __init__(self, frames):
        self._frames = frames

    def history(self, symbol, *, interval="1d", period_days=400):
        if symbol not in self._frames:
            raise ValueError(f"no data for {symbol}")
        return self._frames[symbol]


def test_screen_collects_signals_and_errors(trending_df, monkeypatch):
    frames = {"AAA": trending_df, "BBB": trending_df}
    fake = _FakeProvider(frames)

    import nsetrade.screener.screener as sc
    monkeypatch.setattr(sc, "get_provider", lambda name, conf: fake)

    result = screen(["AAA", "BBB", "MISSING"])
    assert len(result.signals) == 2
    assert "MISSING" in result.errors
    df = result.to_frame()
    assert list(df["score"]) == sorted(df["score"], reverse=True)
