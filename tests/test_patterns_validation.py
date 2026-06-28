"""Tests for new pattern detectors (H&S, wedge) and out-of-sample edge validation."""

import numpy as np
import pandas as pd

from nsetrade.edge import pattern_edge_validated, ValidatedEdge
from nsetrade.patterns.advanced import (
    ADVANCED_DETECTORS,
    detect_head_shoulders,
    detect_wedge,
)


def frame(close, start="2019-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(np.asarray(close, float), index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.005
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.995
    v = pd.Series(np.full(len(close), 1e5), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def test_detectors_registered():
    assert "head_shoulders" in ADVANCED_DETECTORS
    assert "wedge" in ADVANCED_DETECTORS


def test_head_shoulders_detected_on_constructed_series():
    # left shoulder (100), dip, head (120), dip, right shoulder (100), breakdown
    seg = []
    seg += list(np.linspace(70, 100, 20))   # up to left shoulder
    seg += list(np.linspace(100, 85, 12))   # trough 1
    seg += list(np.linspace(85, 120, 16))   # up to head
    seg += list(np.linspace(120, 85, 16))   # trough 2
    seg += list(np.linspace(85, 100, 12))   # up to right shoulder
    seg += list(np.linspace(100, 80, 14))   # breakdown through neckline (~85)
    m = detect_head_shoulders(frame(seg))
    assert m.found
    assert m.direction == "bearish"
    assert m.breakout_level is not None


def test_inverse_head_shoulders_is_bullish():
    seg = []
    seg += list(np.linspace(130, 100, 20))  # down to left shoulder
    seg += list(np.linspace(100, 115, 12))
    seg += list(np.linspace(115, 80, 16))   # head (lowest)
    seg += list(np.linspace(80, 115, 16))
    seg += list(np.linspace(115, 100, 12))  # right shoulder
    seg += list(np.linspace(100, 125, 14))  # breakout up through neckline
    m = detect_head_shoulders(frame(seg))
    assert m.found
    assert m.direction == "bullish"
    assert "Inverse" in m.name


def test_wedge_runs_and_returns_match():
    # falling wedge: converging downward lines, then break up
    n = 80
    mid = np.linspace(120, 95, n)
    amp = np.linspace(12, 2, n)          # narrowing range = converging
    wobble = amp * np.sin(np.arange(n) / 3.0)
    seg = list(mid + wobble) + list(np.linspace(97, 115, 12))  # breakout up
    m = detect_wedge(frame(seg))
    # heuristic — just assert it executes and yields a PatternMatch
    assert m.name in ("Wedge", "Falling Wedge", "Rising Wedge")


def test_validated_edge_shape_and_robustness_flag():
    rng = np.arange(900)
    close = 100 + np.cumsum(np.sin(rng / 11) * 0.8 + 0.02)
    ve = pattern_edge_validated(frame(close), "triangle", forward_bars=15)
    assert isinstance(ve, ValidatedEdge)
    # the three sub-edges exist and occurrence counts reconcile
    assert ve.full.occurrences == ve.in_sample.occurrences + ve.out_sample.occurrences
    assert isinstance(ve.robust, bool)
    assert ve.verdict
    assert "robust" in ve.describe() or "fragile" in ve.describe() or \
        "unproven" in ve.describe() or "no in-sample" in ve.describe()


def test_validated_edge_no_signals_is_unproven():
    flat = frame(np.full(400, 100.0))     # nothing breaks out on a flat line
    ve = pattern_edge_validated(flat, "cup_and_handle")
    assert not ve.robust
    assert ve.full.occurrences == 0
