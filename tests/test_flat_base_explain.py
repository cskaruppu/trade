"""Tests for the Flat Base detector and the pattern explainer."""

import numpy as np
import pandas as pd

from nsetrade.explain import breakout_check, explain_pattern
from nsetrade.patterns.advanced import PatternMatch, detect_flat_base


def frame(close):
    c = pd.Series(np.asarray(close, float))
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    return pd.DataFrame({"open": c.values, "high": (c + 1).values,
                         "low": (c - 1).values, "close": c.values,
                         "volume": [1e5] * len(close)}, index=idx)


def test_flat_base_detects_advance_then_tight_shelf_then_breakout():
    # prior advance 100 -> 130, then a tight shelf ~128-132, then break to 138
    advance = np.linspace(100, 130, 40)
    shelf = 130 + np.sin(np.arange(30)) * 1.5      # ~±1.5 around 130 (tight)
    breakout = np.linspace(131, 138, 5)
    df = frame(np.concatenate([advance, shelf, breakout]))
    m = detect_flat_base(df)
    assert m.found and m.direction == "bullish"
    assert m.status == "breakout"
    assert m.support < m.breakout_level
    assert m.overlays and m.overlays[0]["kind"] == "band"


def test_flat_base_rejects_no_prior_advance():
    # flat the whole way — tight, but no advance into the base → not a flat base
    df = frame(130 + np.sin(np.arange(80)) * 1.5)
    assert not detect_flat_base(df).found


def test_flat_base_rejects_loose_range():
    advance = np.linspace(100, 130, 40)
    wide = 130 + np.sin(np.arange(30)) * 25         # ±25 → far too loose
    df = frame(np.concatenate([advance, wide, np.linspace(131, 140, 5)]))
    assert not detect_flat_base(df).found


def test_explain_pattern_builds_criteria_stop_and_target():
    advance = np.linspace(100, 130, 40)
    shelf = 130 + np.sin(np.arange(30)) * 1.5
    breakout = np.linspace(131, 138, 5)
    df = frame(np.concatenate([advance, shelf, breakout]))
    m = detect_flat_base(df)
    ex = explain_pattern(df, m, key="flat_base")
    assert "shelf" in ex.basis.lower() or "flat" in ex.basis.lower()
    assert ex.criteria                                   # criteria were produced
    labels = {c.label for c in ex.criteria}
    assert "Breakout" in labels and "Base depth" in labels
    assert ex.stop is not None and ex.stop == m.support
    assert ex.target is not None and ex.target > m.breakout_level
    assert "line in the sand" in ex.invalidation


def _match(level):
    return PatternMatch(name="X", found=True, direction="bullish",
                        status="forming", breakout_level=level, support=level * 0.9)


def test_breakout_confirmed_needs_close_above_and_volume():
    close = list(np.linspace(40, 50, 80))
    close[-1] = 55.0                                  # closes above the 54 trigger
    vol = [1e5] * 79 + [5e5]                          # last bar volume surges
    df = frame(close)
    df["volume"] = vol
    bc = breakout_check(df, _match(54.0))
    assert bc.state == "confirmed" and bc.volume_ok is True


def test_breakout_above_but_light_volume_is_flagged():
    close = list(np.linspace(40, 50, 80))
    close[-1] = 55.0
    df = frame(close)
    df["volume"] = [1e5] * 80                          # no surge on the breakout bar
    bc = breakout_check(df, _match(54.0))
    assert bc.state == "volume_light" and bc.volume_ok is False


def test_breakout_approaching_when_just_below_trigger():
    close = list(np.linspace(40, 50, 80))
    close[-1] = 52.5                                   # ~2.8% below 54 → approaching
    df = frame(close)
    bc = breakout_check(df, _match(54.0))
    assert bc.state == "approaching"
    assert bc.pct_to_level < 0


def test_breakout_far_when_well_below():
    df = frame(list(np.linspace(40, 46, 80)))          # ~15% below 54
    bc = breakout_check(df, _match(54.0))
    assert bc.state == "far"


def test_explain_handles_generic_match_without_key():
    df = frame(np.linspace(100, 120, 60))
    m = PatternMatch(name="Something", found=True, direction="bullish",
                     status="forming", breakout_level=120.0, support=110.0,
                     start=df.index[10], end=df.index[-1])
    ex = explain_pattern(df, m)
    assert ex.basis and ex.stop == 110.0
    # forming (not broken out) → Breakout criterion should be False
    bc = next(c for c in ex.criteria if c.label == "Breakout")
    assert bc.ok is False
