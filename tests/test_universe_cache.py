"""Tests for full-universe CSV parsing and the SQLite scan cache."""

import pytest

from nsetrade import universe
from nsetrade.scan_cache import ScanCache

# A trimmed sample of NSE's EQUITY_L.csv (note the space-padded headers).
NSE_CSV = (
    "SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE,"
    " MARKET LOT, ISIN NUMBER, FACE VALUE\n"
    "RELIANCE, Reliance Industries, EQ, 29-NOV-1995, 10, 1, INE002A01018, 10\n"
    "TCS, Tata Consultancy, EQ, 25-AUG-2004, 1, 1, INE467B01029, 1\n"
    "SOMEBOND, Some Bond Series, BE, 01-JAN-2020, 10, 1, INE000X01010, 10\n"
    "NIFTYBEES, Nifty ETF, , 01-JAN-2010, 1, 1, INF200K01180, 1\n"
)


def test_parse_nse_equity_csv():
    syms = universe.parse_nse_equity_csv(NSE_CSV)
    assert "RELIANCE" in syms
    assert "TCS" in syms
    assert "SOMEBOND" in syms          # BE series kept
    assert "NIFTYBEES" in syms         # blank series kept (ETFs/odd rows)
    assert all(s == s.upper() for s in syms)


def test_load_full_universe_and_get_universe(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "CACHE_DIR", tmp_path)
    # not cached yet → helpful error
    with pytest.raises(ValueError, match="not cached"):
        universe.get_universe("nse_all")
    # write a cached file and it loads
    (tmp_path / "nse_equity.csv").write_text(NSE_CSV, encoding="utf-8")
    syms = universe.get_universe("nse_all")
    assert "RELIANCE" in syms
    assert "nse_all" in universe.list_universes()


def test_unknown_universe_raises():
    with pytest.raises(ValueError, match="unknown universe"):
        universe.get_universe("does_not_exist")


def test_scan_cache_roundtrip(tmp_path):
    db = tmp_path / "scans.db"
    cache = ScanCache(db)
    assert cache.latest("opportunities", "nifty50:long") is None

    rows = [{"symbol": "TCS", "score": 4.2}, {"symbol": "INFY", "score": 3.1}]
    cache.save_run("opportunities", "nifty50:long", rows,
                   meta={"scanned": 50}, now=1000.0)
    hit = cache.latest("opportunities", "nifty50:long")
    assert hit["rows"] == rows
    assert hit["meta"]["scanned"] == 50
    assert hit["created_at"] == 1000.0


def test_scan_cache_latest_is_newest(tmp_path):
    cache = ScanCache(tmp_path / "scans.db")
    cache.save_run("opportunities", "u:long", [{"a": 1}], now=1.0)
    cache.save_run("opportunities", "u:long", [{"a": 2}], now=2.0)
    assert cache.latest("opportunities", "u:long")["rows"] == [{"a": 2}]


def test_scan_cache_prune(tmp_path):
    cache = ScanCache(tmp_path / "scans.db")
    for i in range(8):
        cache.save_run("opportunities", "u:long", [{"i": i}], now=float(i))
    deleted = cache.prune(keep_per_key=3)
    assert deleted == 5
    # newest kept
    assert cache.latest("opportunities", "u:long")["rows"] == [{"i": 7}]
    assert len(cache.list_runs()) == 3
