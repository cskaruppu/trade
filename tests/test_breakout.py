"""Tests for the period (new-high) breakout scanner."""

import numpy as np
import pandas as pd

from nsetrade import breakout


def frame(close):
    idx = pd.bdate_range("2021-01-01", periods=len(close))
    c = pd.Series(np.asarray(close, float), index=idx)
    h = c * 1.005
    return pd.DataFrame({"open": c, "high": h, "low": c * 0.995, "close": c,
                         "volume": pd.Series(1e5, index=idx)})


class _FakeProvider:
    def __init__(self, frames):
        self._frames = frames

    def history(self, symbol, **kw):
        if symbol not in self._frames:
            raise ValueError("no data")
        return self._frames[symbol]


def test_period_breakout_new_high():
    # rising to a fresh high on the last bar
    info = breakout.period_breakout(frame(np.linspace(80, 130, 300)),
                                    lookback_days=252)
    assert info.at_new_high
    assert info.pct_from_high <= 0          # close at/above prior high


def test_period_breakout_not_at_high():
    # peaked then pulled back → not a new high
    seg = list(np.linspace(80, 130, 200)) + list(np.linspace(130, 110, 60))
    info = breakout.period_breakout(frame(seg), lookback_days=252)
    assert not info.at_new_high
    assert info.pct_from_high > 0
    assert info.bars_since_high > 0


def test_periods_table():
    assert breakout.PERIODS["3 months"] == 63
    assert breakout.PERIODS["52 weeks"] == 252
    assert breakout.PERIODS["All-time"] is None


def test_scan_breakouts_filters_and_sorts():
    frames = {
        "NEW": frame(np.linspace(80, 140, 300)),                # new high
        "OLD": frame(list(np.linspace(80, 140, 200))
                     + list(np.linspace(140, 100, 80))),        # pulled back
    }
    res, errors = breakout.scan_breakouts(
        ["NEW", "OLD", "MISSING"], period="52 weeks",
        _provider_obj=_FakeProvider(frames))
    syms = [b.symbol for b in res]
    assert syms == ["NEW"]                  # only the one at a new high
    assert "MISSING" in errors
    assert res[0].as_row()["new high"] == "✓"


def test_tolerance_band_includes_near_breakouts():
    # close 1% below the prior high; tol=0.02 should still count it
    seg = list(np.linspace(80, 130, 250)) + [128.0]
    info = breakout.period_breakout(frame(seg), lookback_days=252, tol=0.02)
    assert info.at_new_high
