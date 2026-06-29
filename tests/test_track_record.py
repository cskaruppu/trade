"""Tests for the signal track record — outcome resolution, scorecard, store."""

import numpy as np
import pandas as pd

from nsetrade import track_record as tr
from nsetrade.track_record import TrackedSignal, TrackRecord, resolve_signal, scorecard


def bars(highs, lows, closes, start="2024-01-08"):
    idx = pd.bdate_range(start, periods=len(highs))
    return pd.DataFrame({"high": highs, "low": lows, "close": closes,
                         "open": closes, "volume": [1e5] * len(highs)}, index=idx)


def _sig(side="long", entry=100, target=110, stop=95):
    return TrackedSignal(id=1, symbol="X", pattern="Cup & Handle", side=side,
                         confidence="High", entry=entry, target=target, stop=stop,
                         entry_date="2024-01-05", status="open")


def test_resolve_target_hit():
    fwd = bars([102, 106, 111], [99, 101, 108], [101, 105, 110])  # reaches 110+
    out = resolve_signal(_sig(), fwd, forward_bars=20)
    assert out[0] == "target"
    assert round(out[1], 3) == 0.10           # (110/100 - 1)


def test_resolve_stop_hit():
    fwd = bars([102, 101, 100], [99, 96, 94], [100, 98, 95])      # drops to 94 <= 95
    out = resolve_signal(_sig(), fwd, forward_bars=20)
    assert out[0] == "stop"
    assert round(out[1], 3) == -0.05          # (95/100 - 1)


def test_resolve_stop_first_when_both_touched():
    # a single bar that touches both 110 (target) and 95 (stop) → stop wins
    fwd = bars([111], [94], [100])
    out = resolve_signal(_sig(), fwd, forward_bars=20)
    assert out[0] == "stop"                   # conservative


def test_resolve_timeout():
    # never hits target or stop within the window → timeout at last close
    fwd = bars([103] * 20, [98] * 20, [102] * 20)
    out = resolve_signal(_sig(), fwd, forward_bars=20)
    assert out[0] == "timeout"
    assert round(out[1], 3) == 0.02


def test_resolve_still_open():
    # only 3 bars, no level hit, window not full → None (still open)
    fwd = bars([103, 104, 102], [98, 99, 97], [102, 103, 101])
    assert resolve_signal(_sig(), fwd, forward_bars=20) is None


def test_resolve_short_side():
    s = _sig(side="short", entry=100, target=90, stop=105)
    fwd = bars([101, 98, 95], [99, 92, 89], [100, 95, 90])   # falls to 89 <= 90
    out = resolve_signal(s, fwd, forward_bars=20)
    assert out[0] == "target"
    assert round(out[1], 3) == round(100 / 90 - 1, 3)        # short profit


def test_scorecard_math():
    def rs(outcome, ret):
        return TrackedSignal(1, "X", "P", "long", "High", 100, 110, 95,
                             "2024-01-01", "resolved", outcome, ret, "2024-02-01")
    sigs = [rs("target", 0.10), rs("target", 0.08), rs("stop", -0.05),
            rs("timeout", 0.02), rs("stop", -0.05)]
    sc = scorecard(sigs)
    assert sc["n"] == 5
    assert round(sc["win_rate"], 2) == 0.60          # 3 of 5 positive
    assert round(sc["hit_target_rate"], 2) == 0.40
    assert round(sc["hit_stop_rate"], 2) == 0.40
    # profit factor = (0.10+0.08+0.02) / (0.05+0.05) = 0.20/0.10 = 2.0
    assert round(sc["profit_factor"], 2) == 2.00


def test_log_dedupe_and_evaluate(tmp_path):
    rec = TrackRecord(tmp_path / "tr.db")
    assert rec.log_signal("X", "Cup & Handle", "long", 100, 110, 95,
                          confidence="High", entry_date="2024-01-05") is True
    # same open signal again → skipped
    assert rec.log_signal("X", "Cup & Handle", "long", 100, 110, 95,
                          entry_date="2024-01-06") is False
    assert len(rec.open_signals()) == 1

    # evaluate with an injected fetch that hits the target
    hist = bars([102, 106, 111] + [111] * 20, [99, 101, 108] + [108] * 20,
                [101, 105, 110] + [110] * 20, start="2024-01-08")

    def fetch(sym):
        return hist

    assert rec.evaluate(fetch=fetch, forward_bars=5) == 1
    assert rec.open_signals() == []
    sc = rec.scorecard()
    assert sc["n"] == 1 and sc["hit_target_rate"] == 1.0


def test_scorecard_by_pattern(tmp_path):
    rec = TrackRecord(tmp_path / "tr.db")
    rec.log_signal("A", "Cup & Handle", "long", 100, 110, 95, entry_date="2024-01-01")
    rec.log_signal("B", "Darvas Box", "long", 100, 110, 95, entry_date="2024-01-01")
    by = rec.scorecard(by="pattern")
    assert "Cup & Handle" in by and "Darvas Box" in by
