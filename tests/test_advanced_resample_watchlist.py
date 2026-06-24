"""Tests for resample, watchlist and the advanced structural patterns.

Pattern detectors are heuristic, so these tests build *idealised* shapes and
assert the detector fires with the right direction — they verify the rules are
wired correctly, not that real markets are this clean.
"""

import numpy as np
import pandas as pd
import pytest

from nsetrade import resample, watchlist
from nsetrade.patterns import advanced


def ohlcv(close, start="2021-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(np.asarray(close, dtype=float), index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.005
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.995
    v = pd.Series(np.full(len(close), 100000.0), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


# ---- resample ------------------------------------------------------------

def test_resample_weekly_aggregates():
    daily = ohlcv(np.linspace(100, 200, 60))
    weekly = resample.resample_ohlcv(daily, "weekly")
    assert len(weekly) < len(daily)
    # weekly high >= weekly close everywhere
    assert (weekly["high"] >= weekly["close"]).all()
    # volume is summed -> weekly total >= any daily bar
    assert weekly["volume"].iloc[0] >= 100000.0


def test_resample_daily_passthrough_and_bad_tf():
    daily = ohlcv(np.linspace(100, 110, 20))
    assert resample.resample_ohlcv(daily, "daily") is daily
    with pytest.raises(ValueError):
        resample.resample_ohlcv(daily, "yearly")


def test_scale_period_days():
    assert resample.scale_period_days(400, "monthly") > 400
    assert resample.scale_period_days(400, "daily") == 400


# ---- watchlist -----------------------------------------------------------

def test_watchlist_add_remove(tmp_path):
    p = str(tmp_path / "wl.txt")
    watchlist.add(["reliance", "INFY"], path=p)
    watchlist.add(["tcs", "infy"], path=p)  # dupe ignored, lowercased
    assert watchlist.load(p) == ["RELIANCE", "INFY", "TCS"]
    watchlist.remove(["INFY"], path=p)
    assert watchlist.load(p) == ["RELIANCE", "TCS"]


def test_watchlist_import_nse_csv(tmp_path):
    csv_path = tmp_path / "EQUITY_L.csv"
    csv_path.write_text(
        "SYMBOL,NAME OF COMPANY,SERIES\n"
        "RELIANCE,Reliance Industries,EQ\n"
        "TCS,Tata Consultancy,EQ\n"
    )
    p = str(tmp_path / "wl.txt")
    out = watchlist.import_csv(str(csv_path), path=p, merge=False)
    assert out == ["RELIANCE", "TCS"]


def test_watchlist_import_no_header_single_col(tmp_path):
    csv_path = tmp_path / "list.csv"
    csv_path.write_text("INFY\nWIPRO\n")
    p = str(tmp_path / "wl.txt")
    out = watchlist.import_csv(str(csv_path), path=p, merge=False)
    assert "INFY" in out and "WIPRO" in out


# ---- advanced patterns ---------------------------------------------------

def test_darvas_box_breakout():
    rng = np.random.RandomState(0)
    box = 100 + rng.uniform(-5, 5, 55)          # tight consolidation ~95-105
    breakout = np.linspace(106, 112, 6)          # break above the box top
    df = ohlcv(np.concatenate([box, breakout]))
    m = advanced.detect_darvas_box(df)
    assert m.found and m.direction == "bullish"
    assert m.status == "breakout"
    assert m.breakout_level is not None


def test_cup_and_handle_found():
    x = np.linspace(0, np.pi, 100)
    cup = 100 - 25 * np.sin(x)                    # smooth U: 100 -> 75 -> 100
    handle = np.array([99, 97, 96, 96.5, 97.5])   # mild handle pullback
    df = ohlcv(np.concatenate([cup, handle]))
    m = advanced.detect_cup_and_handle(df)
    assert m.found and m.direction == "bullish"


def test_flag_breakout():
    pole = np.linspace(100, 130, 9)               # +30% pole
    flag = np.array([125, 123, 122, 124, 123])    # shallow consolidation
    end = np.array([128.0])                        # breakout above flag, below pole top
    df = ohlcv(np.concatenate([pole, flag, end]))
    m = advanced.detect_flag(df)
    assert m.found and m.direction == "bullish"
    assert m.status == "breakout"


def test_double_bottom_found():
    close = np.concatenate([
        np.linspace(95, 80, 8),    # down to first bottom
        np.linspace(80, 95, 8),    # up to the neckline peak
        np.linspace(95, 81, 8),    # down to second bottom (~same level)
        np.linspace(81, 98, 8),    # breakout above neckline
    ])
    df = ohlcv(close)
    m = advanced.detect_double_bottom(df)
    assert m.found and m.direction == "bullish"


def test_ascending_triangle_found():
    seg = []
    lows = [85, 88, 91, 93]
    for lo in lows:
        seg.append(np.linspace(lo, 100, 6))      # rise to flat ~100 resistance
        seg.append(np.linspace(100, lo + 3, 6))  # pull back to a higher low
    df = ohlcv(np.concatenate(seg))
    m = advanced.detect_triangle(df)
    assert m.found and m.name == "Ascending Triangle"


def test_detect_advanced_returns_only_found():
    df = ohlcv(np.linspace(100, 101, 60))         # flat-ish: few/no patterns
    matches = advanced.detect_advanced(df, only_found=True)
    assert all(m.found for m in matches)
