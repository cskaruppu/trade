"""Tests for the NSE Bhavcopy parser, store, downloader, splits, and provider.

Network is never touched: fetch is injected, and the provider reads a local
store we populate directly.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nsetrade import bhavcopy as bc

# trimmed sample of NSE's full security-wise bhavcopy (space-padded headers)
SAMPLE = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,"
    " LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS,"
    " NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "RELIANCE, EQ, 27-Jun-2026, 2900, 2910, 2950, 2905, 2940, 2942, 1.2, "
    "5000000, 1000, 50000, 2500000, 50\n"
    "TCS, EQ, 27-Jun-2026, 3800, 3810, 3850, 3805, 3845, 3840, 1.1, "
    "2000000, 800, 40000, 1000000, 50\n"
    "SOMEBOND, GS, 27-Jun-2026, 100, 100, 100, 100, 100, 100, 0, 10, 1, 1, 5, 50\n"
)


def test_bhavcopy_url_format():
    url = bc.bhavcopy_url(dt.date(2026, 6, 27))
    assert url.endswith("sec_bhavdata_full_27062026.csv")


def test_parse_bhavcopy_equity_only():
    rows = bc.parse_bhavcopy(SAMPLE)
    syms = {r["symbol"] for r in rows}
    assert syms == {"RELIANCE", "TCS"}        # GS series excluded
    rel = next(r for r in rows if r["symbol"] == "RELIANCE")
    assert rel["open"] == 2910 and rel["high"] == 2950
    assert rel["close"] == 2942 and rel["volume"] == 5000000   # CLOSE_PRICE col


def test_store_roundtrip_and_history(tmp_path):
    store = bc.BhavcopyStore(tmp_path / "b.db")
    d1, d2 = dt.date(2026, 6, 25), dt.date(2026, 6, 26)
    store.save_day(d1, [{"symbol": "TCS", "open": 10, "high": 11, "low": 9,
                         "close": 10.5, "volume": 100}])
    store.save_day(d2, [{"symbol": "TCS", "open": 10.5, "high": 12, "low": 10,
                         "close": 11.5, "volume": 120}])
    assert store.has_date(d1)
    assert store.symbols() == ["TCS"]
    hist = store.history("TCS")
    assert list(hist["close"]) == [10.5, 11.5]
    assert hist.index[0] == pd.Timestamp("2026-06-25")


def test_download_range_skips_weekends_and_existing(tmp_path):
    store = bc.BhavcopyStore(tmp_path / "b.db")
    calls = []

    def fake_fetch(d):
        calls.append(d)
        return [{"symbol": "X", "open": 1, "high": 1, "low": 1, "close": 1,
                 "volume": 1}]

    # Mon 2026-06-22 .. Sun 2026-06-28 → 5 weekdays fetched, weekend skipped
    summary = bc.download_range(store, dt.date(2026, 6, 22),
                                dt.date(2026, 6, 28), fetch=fake_fetch)
    assert summary["trading_days"] == 5
    assert all(d.weekday() < 5 for d in calls)
    # a second run fetches nothing new (already stored)
    calls.clear()
    bc.download_range(store, dt.date(2026, 6, 22), dt.date(2026, 6, 28),
                      fetch=fake_fetch)
    assert calls == []


def test_adjust_splits_makes_series_continuous():
    # 200 days, then a clean 5:1 split (price /5) for the last 40 days
    pre = np.linspace(900, 1000, 200)
    post = np.linspace(200, 230, 40)        # ~1000 -> ~200 = 5x drop
    close = np.concatenate([pre, post])
    idx = pd.bdate_range("2024-01-01", periods=len(close))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1e5}, index=idx)
    adj = bc.adjust_splits(df)
    # the pre-split prices should be divided by ~5 → no giant gap remains
    ratios = adj["close"].iloc[1:].values / adj["close"].iloc[:-1].values
    assert np.nanmax(np.abs(ratios - 1)) < 0.5      # no ~5x cliff left


def test_provider_serves_from_store(tmp_path):
    store = bc.BhavcopyStore(tmp_path / "b.db")
    idx = pd.bdate_range("2025-01-01", periods=120)
    for i, d in enumerate(idx):
        px = 100 + i * 0.5
        store.save_day(d.date(), [{"symbol": "TCS", "open": px, "high": px + 1,
                                   "low": px - 1, "close": px, "volume": 1000}])
    prov = bc.BhavcopyProvider(db=str(tmp_path / "b.db"), adjust=False)
    hist = prov.history("TCS", period_days=400)
    assert list(hist.columns) == ["open", "high", "low", "close", "volume"]
    assert len(hist) == 120
    with pytest.raises(ValueError, match="no Bhavcopy data"):
        prov.history("MISSING")
