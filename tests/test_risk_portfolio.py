"""Tests for the risk-managed and portfolio backtesters, charts and live scan."""

import numpy as np
import pandas as pd
import pytest

from nsetrade.backtest import backtest_portfolio, backtest_risk


def _wavy(start="2021-01-01", n=400, seed=0):
    """A noisy series with dips, so rsi_ma actually enters trades."""
    rng = np.random.RandomState(seed)
    trend = np.linspace(100, 180, n)
    noise = np.cumsum(rng.randn(n) * 1.5)
    prices = trend + noise + np.sin(np.arange(n) / 8) * 8
    prices = np.maximum(prices, 5)
    idx = pd.bdate_range(start, periods=n)
    c = pd.Series(prices, index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.01
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.99
    v = pd.Series(rng.randint(1e5, 1e6, n), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def test_risk_backtest_metrics_sane():
    df = _wavy(seed=1)
    res = backtest_risk("TEST", df, strategy="rsi_ma",
                        capital=100_000, risk_per_trade=0.01,
                        stop_atr=2.0, target_atr=4.0)
    m = res.metrics
    assert -1.0 <= m["max_drawdown"] <= 0.0
    assert 0.0 <= m["win_rate"] <= 1.0
    assert m["final_equity"] > 0
    assert m["num_trades"] == len(res.trades)
    assert "Sharpe" in res.summary()
    # every exit reason is one of the allowed kinds
    assert set(m["exit_counts"]).issubset({"stop", "target", "signal", "eod"})


def test_risk_stop_caps_loss_per_trade():
    df = _wavy(seed=2)
    res = backtest_risk("TEST", df, strategy="breakout",
                        capital=100_000, risk_per_trade=0.01, stop_atr=2.0)
    # each completed trade's loss should be bounded near the risk budget.
    # Allow slack for gap-throughs and intrabar fills.
    for t in res.trades:
        loss = -t.pnl
        if loss > 0:
            assert loss <= 0.05 * 100_000  # well under 5% of capital per trade


def test_risk_no_target_uses_none():
    df = _wavy(seed=3)
    res = backtest_risk("TEST", df, strategy="rsi_ma", target_atr=None)
    assert "target" not in res.metrics["exit_counts"]


def test_portfolio_aggregates():
    frames = {"AAA": _wavy(seed=4), "BBB": _wavy(seed=5), "CCC": _wavy(seed=6)}
    res = backtest_portfolio(frames, strategy="rsi_ma", capital=900_000)
    m = res.metrics
    assert m["num_symbols"] == 3
    assert len(res.per_symbol) == 3
    assert res.equity_curve.iloc[-1] > 0
    assert -1.0 <= m["max_drawdown"] <= 0.0
    assert "Portfolio" in res.summary()


def test_portfolio_skips_bad_symbol():
    short = _wavy(n=10)  # too short -> backtest_risk raises, should be skipped
    frames = {"GOOD": _wavy(seed=7), "BAD": short}
    res = backtest_portfolio(frames, strategy="rsi_ma")
    assert "GOOD" in res.per_symbol
    assert "BAD" in res.errors


def test_chart_renders_png(tmp_path):
    df = _wavy(seed=8)
    from nsetrade.charts import render_chart

    out = render_chart("TEST", df, out_path=str(tmp_path / "t.png"), bars=120)
    assert (tmp_path / "t.png").exists()
    assert (tmp_path / "t.png").stat().st_size > 1000


# ---- live scanning logic -------------------------------------------------

def test_is_market_open():
    import datetime as dt
    from nsetrade.live import IST, is_market_open

    # Monday 11:00 IST -> open
    mon = dt.datetime(2026, 6, 8, 11, 0, tzinfo=IST)
    assert is_market_open(mon)
    # Monday 08:00 IST -> closed (pre-open)
    assert not is_market_open(dt.datetime(2026, 6, 8, 8, 0, tzinfo=IST))
    # Saturday -> closed
    assert not is_market_open(dt.datetime(2026, 6, 13, 11, 0, tzinfo=IST))


def test_diff_alerts_detects_new_buy():
    from nsetrade.live import diff_alerts
    from nsetrade.signals.engine import Signal
    import pandas as pd

    def sig(verdict, score, reasons):
        return Signal("AAA", pd.Timestamp("2026-01-01"), 100.0, score,
                      verdict, reasons)

    prev = {"AAA": sig("Neutral", 0.0, [])}
    curr = {"AAA": sig("Buy", 2.0, ["Golden Cross (50/200)"])}
    alerts = diff_alerts(prev, curr)
    assert len(alerts) == 1
    assert alerts[0].kind == "verdict_change"
    assert "🟢" in alerts[0].line()

    # no change -> no alert
    assert diff_alerts(curr, curr) == []
