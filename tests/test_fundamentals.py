"""Tests for fundamentals parsing, debt classification, and news normalization."""

from nsetrade import fundamentals as fnd


def test_debt_status_thresholds():
    assert fnd.debt_status(0, None) == "Debt-free ✓"
    assert fnd.debt_status(None, 5) == "Near debt-free"
    assert fnd.debt_status(None, 30) == "Low debt"
    assert fnd.debt_status(None, 80) == "Moderate debt"
    assert fnd.debt_status(None, 150) == "High debt"
    assert fnd.debt_status(None, None) == "Debt: n/a"


def test_parse_fundamentals_full():
    info = {
        "longName": "Reliance Industries", "sector": "Energy",
        "industry": "Oil & Gas", "marketCap": 18_00_000 * 1e7,  # 18 lakh Cr
        "trailingPE": 24.5, "debtToEquity": 35.0, "totalDebt": 3.0e12,
        "returnOnEquity": 0.12, "profitMargins": 0.09, "dividendYield": 0.004,
        "fiftyTwoWeekHigh": 1600.0, "fiftyTwoWeekLow": 1100.0,
    }
    f = fnd.parse_fundamentals("RELIANCE", info)
    assert f.name == "Reliance Industries"
    assert f.sector == "Energy"
    assert f.pe == 24.5
    assert f.debt_status == "Low debt"
    assert any("Sector: Energy" in h for h in f.highlights)
    assert any("P/E: 24.5" in h for h in f.highlights)
    assert "Low debt" in f.summary()
    assert "P/E 24.5" in f.summary()


def test_parse_fundamentals_debt_free():
    info = {"longName": "X", "totalDebt": 0, "marketCap": 5000 * 1e7}
    f = fnd.parse_fundamentals("X", info)
    assert f.debt_status == "Debt-free ✓"
    assert f.highlights[0] == "Debt-free ✓"


def test_parse_fundamentals_missing_fields():
    f = fnd.parse_fundamentals("Y", {})
    assert f.pe is None and f.market_cap is None
    assert f.debt_status == "Debt: n/a"
    assert f.summary()  # still returns something


def test_normalize_news_old_shape():
    item = {"title": "Q3 results beat", "link": "http://x", "publisher": "ET",
            "providerPublishTime": 1700000000}
    n = fnd._normalize_news(item)
    assert n["title"] == "Q3 results beat"
    assert n["link"] == "http://x"
    assert n["publisher"] == "ET"


def test_normalize_news_new_shape():
    item = {"content": {"title": "Stock surges", "provider": {"displayName": "Mint"},
                        "canonicalUrl": {"url": "http://y"}, "pubDate": "2026-01-01"}}
    n = fnd._normalize_news(item)
    assert n["title"] == "Stock surges"
    assert n["link"] == "http://y"
    assert n["publisher"] == "Mint"
