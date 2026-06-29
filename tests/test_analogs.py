"""Tests for the historical chart-analog matcher."""

import numpy as np
import pandas as pd

from nsetrade.analogs import find_analogs


def frame(close):
    c = pd.Series(np.asarray(close, float))
    idx = pd.bdate_range("2010-01-01", periods=len(close))
    return pd.DataFrame({"open": c.values, "high": (c + 1).values,
                         "low": (c - 1).values, "close": c.values,
                         "volume": [1e5] * len(close)}, index=idx)


def test_not_enough_history():
    res = find_analogs(frame(np.linspace(100, 120, 30)), window=40, forward=30)
    assert res.n == 0
    assert "not enough history" in res.note


def test_repeated_shape_is_found_with_high_similarity():
    # A distinctive V-shape that repeats, each followed by a strong rally.
    rng = np.random.RandomState(0)
    noise = lambda k: rng.normal(0, 0.3, k)
    v = np.concatenate([np.linspace(100, 80, 20), np.linspace(80, 110, 20)])
    rally = np.linspace(110, 140, 30)
    flat = np.full(60, 100.0) + noise(60)
    # episode 1: V then rally, then filler, then the *current* V at the end
    closes = np.concatenate([
        flat,                       # 0-59  unrelated
        v + noise(40),              # 60-99 past V
        rally + noise(30),          # 100-129 forward rally after past V
        flat,                       # 130-189 unrelated
        v + noise(40),              # 190-229 current V (last window)
    ])
    df = frame(closes)
    res = find_analogs(df, window=40, forward=30, min_similarity=0.7)
    assert res.n >= 1
    # the past V (ending around index 99) should surface as an analog
    assert max(a.similarity for a in res.analogs) > 0.8
    # and its forward return was strongly positive (the rally)
    assert res.avg_forward > 0
    assert "analogs" in res.describe()


def test_no_match_when_history_is_unrelated_noise():
    rng = np.random.RandomState(1)
    closes = 100 + np.cumsum(rng.normal(0, 1, 300))
    res = find_analogs(frame(closes), window=40, forward=30, min_similarity=0.98)
    # an extremely strict threshold should yield no analogs
    assert res.n == 0


def test_analogs_are_non_overlapping():
    rng = np.random.RandomState(2)
    closes = 100 + np.cumsum(rng.normal(0, 1, 600))
    res = find_analogs(frame(closes), window=40, forward=30,
                       min_similarity=0.0, top_k=8, min_gap=40)
    dates = [a.end_date for a in res.analogs]
    assert len(dates) == len(set(dates))   # no duplicate episodes
