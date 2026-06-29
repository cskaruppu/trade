"""Tests for new pattern detectors (H&S, wedge) and out-of-sample edge validation."""

import numpy as np
import pandas as pd

from nsetrade.edge import pattern_edge_validated, ValidatedEdge
from nsetrade.patterns.advanced import (
    ADVANCED_DETECTORS,
    detect_head_shoulders,
    detect_vcp,
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
    assert "vcp" in ADVANCED_DETECTORS
    assert "accumulation" in ADVANCED_DETECTORS


def test_accumulation_base_detected():
    from nsetrade.patterns.advanced import detect_accumulation
    rng = np.random.RandomState(1)
    down = list(np.linspace(300, 228, 55) + rng.randn(55) * 3)
    base = [234 - i * 0.05 + np.sin(i / 3) * 8 + rng.randn() * 2 for i in range(60)]
    seg = down + base
    m = detect_accumulation(frame(seg))
    assert m.found
    assert m.direction == "bullish"
    assert m.breakout_level is not None        # the falling-resistance level
    assert m.support is not None               # the support zone
    kinds = [o["kind"] for o in (m.overlays or [])]
    assert "line" in kinds and "band" in kinds  # trendline + support band


def test_accumulation_rejects_uptrend():
    from nsetrade.patterns.advanced import detect_accumulation
    # a clean uptrend has no prior decline + flat base → not accumulation
    m = detect_accumulation(frame(np.linspace(100, 220, 130)))
    assert not m.found


def test_vcp_detected_on_contracting_base():
    # progressively tighter pullbacks: 25% -> 12% -> 5%, then breakout
    seg = []
    seg += list(np.linspace(50, 100, 25))    # rally
    seg += list(np.linspace(100, 75, 12))    # pullback 1 (~25%)
    seg += list(np.linspace(75, 105, 16))    # rally to new high
    seg += list(np.linspace(105, 92, 10))    # pullback 2 (~12%)
    seg += list(np.linspace(92, 108, 12))    # rally
    seg += list(np.linspace(108, 102, 8))    # pullback 3 (~5%)
    seg += list(np.linspace(102, 116, 10))   # breakout above pivot
    m = detect_vcp(frame(seg))
    assert m.found
    assert m.direction == "bullish"
    assert m.breakout_level is not None
    assert "contractions" in m.note


def test_cup_overlays_trace_the_shape():
    from nsetrade.patterns.advanced import detect_cup_and_handle
    n = 180
    close = np.zeros(n)
    close[:10] = np.linspace(298, 300, 10)
    close[10:90] = 300 - 90 * np.sin(np.linspace(0, np.pi / 2, 80))
    close[90:165] = 210 + 88 * np.sin(np.linspace(0, np.pi / 2, 75))
    close[165:] = np.linspace(298, 290, 15)
    m = detect_cup_and_handle(frame(close))
    assert m.found
    assert m.overlays and len(m.overlays) == 2
    curve, rim = m.overlays
    assert curve["kind"] == "spline"
    # the cup curve dips to the bottom then returns to the rim
    assert min(curve["y"]) < 230 < max(curve["y"])
    # dates are real timestamps (not the 1970 epoch bug)
    assert curve["x"][0].year >= 2019
    assert rim["kind"] == "line" and len(rim["y"]) == 2


def test_overlays_render_in_chart_without_error():
    from nsetrade.charts_interactive import build_figure
    n = 180
    close = np.concatenate([
        np.linspace(298, 300, 10),
        300 - 90 * np.sin(np.linspace(0, np.pi / 2, 80)),
        210 + 88 * np.sin(np.linspace(0, np.pi / 2, 75)),
        np.linspace(298, 290, 15)])
    fig, notes = build_figure("TEST", frame(close), bars=200, show_patterns=True)
    assert len(fig.data) > 0


def test_measured_move_target_drawn_on_breakout():
    from nsetrade.charts_interactive import build_figure
    # cup that breaks OUT above the rim → a measured-move target should appear
    close = np.concatenate([
        np.linspace(298, 300, 10),
        300 - 90 * np.sin(np.linspace(0, np.pi / 2, 80)),
        210 + 88 * np.sin(np.linspace(0, np.pi / 2, 75)),
        np.linspace(298, 308, 15)])           # breakout above the rim
    fig, notes = build_figure("TEST", frame(close), bars=200, show_patterns=True)
    assert any("Upside potential" in n for n in notes)


def test_volume_confirms_detects_surge():
    from nsetrade.patterns.advanced import volume_confirms
    base = frame(np.linspace(90, 110, 80))
    base["volume"] = 1000.0
    # latest bar volume spikes → confirmed
    base.iloc[-1, base.columns.get_loc("volume")] = 5000.0
    assert volume_confirms(base, lookback=50, mult=1.3) is True
    # flat volume → not confirmed
    flat = frame(np.linspace(90, 110, 80))
    flat["volume"] = 1000.0
    assert volume_confirms(flat, lookback=50, mult=1.3) is False


def test_vcp_rejects_widening_pullbacks():
    # pullbacks getting DEEPER (not a VCP)
    seg = []
    seg += list(np.linspace(50, 100, 25))
    seg += list(np.linspace(100, 95, 8))     # 5%
    seg += list(np.linspace(95, 108, 14))
    seg += list(np.linspace(108, 90, 10))    # ~17% (widening)
    seg += list(np.linspace(90, 112, 14))
    seg += list(np.linspace(112, 80, 12))    # ~29% (widening more)
    m = detect_vcp(frame(seg))
    assert not m.found


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
