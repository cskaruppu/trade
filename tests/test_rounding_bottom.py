"""Tests for the Rounding Bottom / saucer detector."""

import numpy as np
import pandas as pd

from nsetrade.explain import explain_pattern
from nsetrade.patterns.advanced import detect_cup_and_handle, detect_rounding_bottom


def frame(close):
    c = pd.Series(np.asarray(close, float))
    idx = pd.bdate_range("2019-01-01", periods=len(close))
    return pd.DataFrame({"open": c.values, "high": (c + 2).values,
                         "low": (c - 2).values, "close": c.values,
                         "volume": [1e5] * len(close)}, index=idx)


def _saucer(rim=490.0, bottom=150.0, n=120):
    """A deep, rounded U from rim down to bottom and back, reclaiming the rim."""
    half = n // 2
    xs = np.linspace(-1, 1, 2 * half)
    u = bottom + (rim - bottom) * (xs ** 2)     # parabola: rim at ends, bottom mid
    return u


def test_rounding_bottom_detects_deep_saucer_breakout():
    u = _saucer()
    closes = np.concatenate([u, np.linspace(490, 520, 6)])   # break above the rim
    df = frame(closes)
    m = detect_rounding_bottom(df)
    assert m.found and m.direction == "bullish"
    assert m.status == "breakout"
    depth = (m.breakout_level - m.support) / m.breakout_level
    assert 0.35 <= depth <= 0.80                              # genuinely deep
    assert m.overlays and m.overlays[0]["kind"] == "spline"


def test_rounding_bottom_rejects_shallow_base():
    # a shallow ~20% U is a cup, not a rounding bottom
    u = _saucer(rim=220.0, bottom=180.0)
    df = frame(np.concatenate([u, np.linspace(220, 230, 6)]))
    assert not detect_rounding_bottom(df).found


def test_rounding_bottom_rejects_sharp_v():
    # a sharp V spends almost no time near the low → not a saucer
    down = np.linspace(490, 150, 8)
    up = np.linspace(150, 500, 8)
    pad = np.full(60, 490.0)
    df = frame(np.concatenate([pad, down, up, pad]))
    assert not detect_rounding_bottom(df).found


def test_deep_saucer_is_not_misread_as_a_cup():
    # the whole point: a 69%-deep base is too deep for the cup detector
    u = _saucer(rim=490.0, bottom=150.0)
    df = frame(np.concatenate([u, np.linspace(490, 520, 6)]))
    assert not detect_cup_and_handle(df).found        # cup rejects >50% depth
    assert detect_rounding_bottom(df).found           # rounding bottom catches it


def test_explain_rounding_bottom_basis():
    u = _saucer()
    df = frame(np.concatenate([u, np.linspace(490, 520, 6)]))
    m = detect_rounding_bottom(df)
    ex = explain_pattern(df, m, key="rounding_bottom")
    assert "saucer" in ex.basis.lower() or "rounding" in ex.basis.lower()
    assert ex.target is not None and ex.stop is not None
