"""Tests for the standalone support/resistance trendline fitter."""

import numpy as np
import pandas as pd

from nsetrade.trendlines import fit_trendlines


def frame(close):
    c = pd.Series(np.asarray(close, float))
    idx = pd.bdate_range("2025-01-01", periods=len(close))
    return pd.DataFrame({"open": c.values, "high": (c + 3).values,
                         "low": (c - 3).values, "close": c.values,
                         "volume": [1e5] * len(close)}, index=idx)


def test_downtrend_resistance_slopes_down():
    df = frame(np.linspace(200, 140, 120) + np.sin(np.arange(120) / 9) * 6)
    tl = fit_trendlines(df)
    assert "resistance" in tl and "support" in tl
    assert tl["resistance"]["slope"] < 0
    assert tl["resistance"]["direction"] == "down"
    # endpoints are two dates / two prices
    assert len(tl["resistance"]["x"]) == 2 and len(tl["resistance"]["y"]) == 2


def test_uptrend_support_slopes_up():
    # gentler rise + bigger oscillation so swing pivots are clearly detectable
    df = frame(np.linspace(100, 160, 120) + np.sin(np.arange(120) / 6) * 12)
    tl = fit_trendlines(df)
    assert "support" in tl
    assert tl["support"]["slope"] > 0
    assert tl["support"]["direction"] == "up"


def test_too_few_bars_returns_empty():
    assert fit_trendlines(frame(np.linspace(100, 110, 10))) == {}
