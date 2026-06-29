"""Tests for the focused pattern screener (Cup & Handle / Darvas Box)."""

import numpy as np
import pandas as pd

from nsetrade import pattern_scan
from nsetrade.patterns.advanced import PatternMatch


def frame(close):
    idx = pd.bdate_range("2021-01-01", periods=len(close))
    c = pd.Series(np.asarray(close, float), index=idx)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": pd.Series(1e5, index=idx)})


class _FakeProvider:
    def __init__(self, frames):
        self._frames = frames

    def history(self, symbol, **kw):
        if symbol not in self._frames:
            raise ValueError("no data")
        return self._frames[symbol]


def test_scan_filters_to_requested_patterns(monkeypatch):
    # make detect_advanced return a cup and a flag regardless of data
    fake = [
        PatternMatch(name="Cup & Handle", found=True, direction="bullish",
                     status="breakout", breakout_level=105.0),
        PatternMatch(name="Bull Flag", found=True, direction="bullish",
                     status="forming", breakout_level=110.0),
    ]
    monkeypatch.setattr("nsetrade.patterns.detect_advanced", lambda df: fake)
    prov = _FakeProvider({"AAA": frame(np.linspace(90, 110, 200))})

    hits, errors = pattern_scan.scan_for_patterns(
        ["AAA"], ["cup_and_handle", "darvas_box"], with_edge=False,
        _provider_obj=prov)
    # only the cup matches the requested keys; the flag is filtered out
    assert [h.pattern for h in hits] == ["Cup & Handle"]
    assert hits[0].as_row()["symbol"] == "AAA"


def test_only_breakouts_filter(monkeypatch):
    fake = [PatternMatch(name="Darvas Box", found=True, direction="bullish",
                         status="forming", breakout_level=100.0)]
    monkeypatch.setattr("nsetrade.patterns.detect_advanced", lambda df: fake)
    prov = _FakeProvider({"AAA": frame(np.linspace(90, 110, 200))})

    hits, _ = pattern_scan.scan_for_patterns(
        ["AAA"], ["darvas_box"], with_edge=False, only_breakouts=True,
        _provider_obj=prov)
    assert hits == []          # forming, and we asked for breakouts only


def test_errors_collected(monkeypatch):
    monkeypatch.setattr("nsetrade.patterns.detect_advanced", lambda df: [])
    prov = _FakeProvider({})
    hits, errors = pattern_scan.scan_for_patterns(
        ["MISSING"], with_edge=False, _provider_obj=prov)
    assert hits == []
    assert "MISSING" in errors


def test_as_row_shape(monkeypatch):
    fake = [PatternMatch(name="Cup & Handle", found=True, direction="bullish",
                         status="breakout", breakout_level=105.0, note="depth 20%")]
    monkeypatch.setattr("nsetrade.patterns.detect_advanced", lambda df: fake)
    prov = _FakeProvider({"AAA": frame(np.linspace(90, 110, 200))})
    hits, _ = pattern_scan.scan_for_patterns(["AAA"], with_edge=False,
                                             _provider_obj=prov)
    row = hits[0].as_row()
    assert set(["symbol", "pattern", "status", "vol", "close", "breakout",
                "edge", "robust"]) <= row.keys()
    assert row["edge"] == "-"        # edge disabled


def test_measured_move_target_and_upside(monkeypatch):
    # breakout at 110, support 90 → target 130, close 100 → +30% upside
    fake = [PatternMatch(name="Cup & Handle", found=True, direction="bullish",
                         status="breakout", breakout_level=110.0, support=90.0)]
    monkeypatch.setattr("nsetrade.patterns.detect_advanced", lambda df: fake)
    f = frame(np.full(200, 100.0))     # close = 100
    prov = _FakeProvider({"AAA": f})
    hits, _ = pattern_scan.scan_for_patterns(["AAA"], ["cup_and_handle"],
                                             with_edge=False, _provider_obj=prov)
    h = hits[0]
    assert h.target == 130.0                       # 110 + (110 - 90)
    assert round(h.upside_pct, 2) == 0.30          # (130/100 - 1)
    row = h.as_row()
    assert row["target"] == 130.0
    assert row["upside %"] == 30.0


def test_only_volume_confirmed_filter(monkeypatch):
    fake = [
        PatternMatch(name="Cup & Handle", found=True, direction="bullish",
                     status="breakout", breakout_level=105.0, volume_confirmed=True),
        PatternMatch(name="Darvas Box", found=True, direction="bullish",
                     status="breakout", breakout_level=100.0, volume_confirmed=False),
    ]
    monkeypatch.setattr("nsetrade.patterns.detect_advanced", lambda df: fake)
    prov = _FakeProvider({"AAA": frame(np.linspace(90, 110, 200))})
    hits, _ = pattern_scan.scan_for_patterns(
        ["AAA"], ["cup_and_handle", "darvas_box"], with_edge=False,
        only_volume_confirmed=True, _provider_obj=prov)
    assert [h.pattern for h in hits] == ["Cup & Handle"]   # the vol-confirmed one
    assert hits[0].as_row()["vol"] == "✓"
