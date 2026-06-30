"""Tests for the holding-period outlook (horizon-conditioned base rates)."""

import numpy as np
import pandas as pd

from nsetrade.benchmarks import benchmark_for
from nsetrade.outlook import holding_period_outlook


def frame(close):
    c = pd.Series(np.asarray(close, float))
    idx = pd.bdate_range("2008-01-01", periods=len(close))
    return pd.DataFrame({"open": c.values, "high": (c + 1).values,
                         "low": (c - 1).values, "close": c.values,
                         "volume": [1e5] * len(close)}, index=idx)


def test_not_enough_history():
    out = holding_period_outlook(frame(np.linspace(100, 110, 50)), window=40)
    assert not out.horizons
    assert "not enough" in out.note


def test_uptrend_outlook_is_positive_with_bands():
    # steady compounding uptrend with mild noise → forward returns positive
    rng = np.random.RandomState(0)
    t = np.arange(1500)
    closes = 100 * (1.0008 ** t) * (1 + rng.normal(0, 0.01, t.size))
    out = holding_period_outlook(frame(closes), window=40,
                                 horizons={"3M": 63, "6M": 126, "1Y": 252},
                                 min_similarity=0.0, top_k=40)
    assert out.n_analogs > 0
    labels = [h.label for h in out.horizons]
    assert labels == ["3M", "6M", "1Y"]            # sorted by horizon length
    for h in out.horizons:
        assert h.n > 0
        assert h.p25 <= h.median <= h.p75           # band ordering
        assert h.target_low <= h.target_med <= h.target_high
        assert h.median > 0                          # uptrend → positive base rate
    # longer horizon should not have a larger sample than the shortest
    assert out.horizons[-1].n <= out.horizons[0].n


def test_target_band_anchored_to_last_price():
    rng = np.random.RandomState(1)
    closes = 100 + np.cumsum(rng.normal(0.05, 1.0, 1500))
    out = holding_period_outlook(frame(closes), window=40, min_similarity=0.0)
    if out.horizons:
        h = out.horizons[0]
        assert abs(h.target_med - out.price * (1 + h.median)) < 1e-6


def test_benchmark_lookup():
    b = benchmark_for("cup_and_handle")
    assert b and b["reliability"] == "High"
    assert benchmark_for("nonsense_key") is None
    assert benchmark_for(None) is None
