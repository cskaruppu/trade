"""Tests for auto Fibonacci retracement and trend-based extension."""

import numpy as np
import pandas as pd

from nsetrade.fibonacci import (
    RETRACEMENT_RATIOS,
    fib_extension,
    fib_retracement,
)


def frame(close, start="2021-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(np.asarray(close, float), index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.003
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.997
    v = pd.Series(np.full(len(close), 1e5), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def test_retracement_uptrend_levels_are_support():
    # rally from 100 to 200, then a pullback → up-swing, levels are support
    seg = list(np.linspace(100, 200, 60)) + list(np.linspace(200, 160, 20))
    fr = fib_retracement(frame(seg), lookback=100)
    assert fr.found
    assert fr.direction == "up"
    assert fr.swing_low < fr.swing_high
    # 7 standard levels in order from high(0%) to low(100%)
    assert len(fr.levels) == len(RETRACEMENT_RATIOS)
    by = {round(l.ratio, 3): l.price for l in fr.levels}
    assert abs(by[0.0] - fr.swing_high) < 1e-6        # 0% == swing high
    assert abs(by[1.0] - fr.swing_low) < 1e-6         # 100% == swing low
    # 50% retrace sits midway
    assert abs(by[0.5] - (fr.swing_high + fr.swing_low) / 2) < 1e-6
    assert fr.nearest is not None


def test_retracement_downtrend_is_resistance():
    seg = list(np.linspace(200, 100, 60)) + list(np.linspace(100, 130, 20))
    fr = fib_retracement(frame(seg), lookback=100)
    assert fr.found
    assert fr.direction == "down"
    by = {round(l.ratio, 3): l.price for l in fr.levels}
    assert abs(by[0.0] - fr.swing_low) < 1e-6         # 0% == swing low (up from here)
    assert abs(by[1.0] - fr.swing_high) < 1e-6


def test_retracement_too_few_bars():
    fr = fib_retracement(frame(np.linspace(100, 110, 10)))
    assert not fr.found


def test_extension_projects_targets_above_b():
    # A(100) → B(180) → C(150) pullback, then drift
    seg = []
    seg += list(np.linspace(100, 100, 5))
    seg += list(np.linspace(100, 180, 40))    # A up to B
    seg += list(np.linspace(180, 150, 20))    # pull back to C
    seg += list(np.linspace(150, 165, 15))
    ext = fib_extension(frame(seg), lookback=120)
    assert ext.found
    assert ext.point_b > ext.point_a
    # the 1.618 target should sit above B (it's an upside projection)
    by = {round(l.ratio, 3): l.price for l in ext.levels}
    assert by[1.618] > ext.point_b
    assert "targets" in ext.describe()


def test_extension_handles_no_structure():
    ext = fib_extension(frame(np.full(60, 100.0)))
    # flat line → no usable A-B swing
    assert not ext.found
