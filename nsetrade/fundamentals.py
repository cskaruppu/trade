"""Lightweight fundamentals + news for a stock (best-effort, via Yahoo).

The pattern engine is purely technical; this adds the *context* a trader glances
at before acting — is it debt-free, what sector, rough valuation, and recent
headlines. Data comes from Yahoo Finance through yfinance (free, no key), so it
is **unofficial and can be incomplete or stale**, especially for small-caps.
Treat it as a quick reference, not a fundamental analysis.

Parsing (:func:`parse_fundamentals`, :func:`debt_status`) is pure and tested;
the network fetch degrades gracefully to ``None``/empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

CRORE = 1e7


@dataclass
class Fundamentals:
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None        # INR
    pe: Optional[float] = None
    debt_to_equity: Optional[float] = None     # yfinance form (~D/E * 100)
    total_debt: Optional[float] = None
    roe: Optional[float] = None                # fraction
    profit_margin: Optional[float] = None      # fraction
    dividend_yield: Optional[float] = None     # fraction
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    debt_status: str = "Debt: n/a"
    highlights: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary for feeding into the AI context."""
        bits = [self.debt_status]
        if self.sector:
            bits.append(f"sector {self.sector}")
        if self.pe is not None:
            bits.append(f"P/E {self.pe:.1f}")
        if self.roe is not None:
            bits.append(f"ROE {self.roe:.0%}")
        if self.market_cap:
            bits.append(f"mcap ₹{self.market_cap / CRORE:,.0f} Cr")
        return "; ".join(bits)


def _num(info: dict, *keys):
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def debt_status(total_debt: Optional[float], dte: Optional[float]) -> str:
    """Classify leverage from total debt and debt-to-equity (yfinance form)."""
    if total_debt is not None and total_debt == 0:
        return "Debt-free ✓"
    if dte is not None:
        if dte < 10:
            return "Near debt-free"
        if dte < 50:
            return "Low debt"
        if dte < 100:
            return "Moderate debt"
        return "High debt"
    if total_debt is not None and total_debt < 1e6:
        return "Near debt-free"
    return "Debt: n/a"


def parse_fundamentals(symbol: str, info: dict) -> Fundamentals:
    """Build a :class:`Fundamentals` from a Yahoo ``info`` dict. Pure."""
    mcap = _num(info, "marketCap")
    pe = _num(info, "trailingPE", "forwardPE")
    dte = _num(info, "debtToEquity")
    total_debt = _num(info, "totalDebt")
    roe = _num(info, "returnOnEquity")
    margin = _num(info, "profitMargins")
    dy = _num(info, "dividendYield")
    hi = _num(info, "fiftyTwoWeekHigh")
    lo = _num(info, "fiftyTwoWeekLow")
    status = debt_status(total_debt, dte)

    highlights = [status]
    sector = info.get("sector")
    if sector:
        highlights.append(f"Sector: {sector}")
    if mcap:
        highlights.append(f"Market cap: ₹{mcap / CRORE:,.0f} Cr")
    if pe is not None:
        highlights.append(f"P/E: {pe:.1f}")
    if roe is not None:
        highlights.append(f"ROE: {roe:.0%}")
    if margin is not None:
        highlights.append(f"Net margin: {margin:.0%}")
    if dy:
        highlights.append(f"Dividend yield: {dy:.1%}")
    if hi and lo:
        highlights.append(f"52-wk range: ₹{lo:,.0f}–₹{hi:,.0f}")

    return Fundamentals(
        symbol=symbol, name=info.get("longName") or info.get("shortName"),
        sector=sector, industry=info.get("industry"), market_cap=mcap, pe=pe,
        debt_to_equity=dte, total_debt=total_debt, roe=roe, profit_margin=margin,
        dividend_yield=dy, week52_high=hi, week52_low=lo, debt_status=status,
        highlights=highlights)


def _normalize_news(item: dict) -> dict:
    """Normalize a yfinance news item (handles old + new yfinance shapes)."""
    if isinstance(item.get("content"), dict):     # newer yfinance
        c = item["content"]
        url = (c.get("canonicalUrl") or c.get("clickThroughUrl") or {})
        return {"title": c.get("title"),
                "link": url.get("url") if isinstance(url, dict) else None,
                "publisher": (c.get("provider") or {}).get("displayName"),
                "time": c.get("pubDate")}
    return {"title": item.get("title"), "link": item.get("link"),
            "publisher": item.get("publisher"),
            "time": item.get("providerPublishTime")}


def fetch_fundamentals(symbol: str, *, suffix: str = ".NS") -> Optional[Fundamentals]:
    """Fetch fundamentals from Yahoo. Returns ``None`` on any failure (best-effort)."""
    try:
        import yfinance
        info = yfinance.Ticker(symbol.upper() + suffix).info
        if not info:
            return None
        return parse_fundamentals(symbol.upper(), info)
    except Exception:  # noqa: BLE001 - network/parse best-effort
        return None


def fetch_news(symbol: str, *, suffix: str = ".NS", limit: int = 6) -> list[dict]:
    """Fetch recent news headlines from Yahoo. Returns ``[]`` on failure."""
    try:
        import yfinance
        items = yfinance.Ticker(symbol.upper() + suffix).news or []
        out = [_normalize_news(i) for i in items[:limit]]
        return [dict(n, source="Yahoo") for n in out if n.get("title")]
    except Exception:  # noqa: BLE001
        return []


def parse_google_news_rss(xml_text: str, *, limit: int = 8) -> list[dict]:
    """Parse a Google News RSS feed into news items. Pure (stdlib only).

    Google News aggregates many publishers (ET, Moneycontrol, Mint, Reuters…),
    so one feed gives broad coverage. Item titles are usually "Headline - Source".
    """
    import xml.etree.ElementTree as ET

    out: list[dict] = []
    root = ET.fromstring(xml_text)
    for it in root.findall(".//item")[:limit]:
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "link": (it.findtext("link") or "").strip(),
            "publisher": (it.findtext("source") or "Google News").strip(),
            "time": (it.findtext("pubDate") or "").strip(),
            "source": "Google",
        })
    return out


def fetch_google_news(query: str, *, limit: int = 8) -> list[dict]:
    """Fetch recent India-focused news for ``query`` from Google News RSS."""
    import urllib.parse
    import urllib.request

    try:
        q = urllib.parse.quote(query)
        url = (f"https://news.google.com/rss/search?q={q}"
               f"&hl=en-IN&gl=IN&ceid=IN:en")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            xml_text = resp.read().decode("utf-8", "replace")
        return parse_google_news_rss(xml_text, limit=limit)
    except Exception:  # noqa: BLE001 - best-effort
        return []


def fetch_all_news(symbol: str, *, name: Optional[str] = None,
                   suffix: str = ".NS", limit: int = 8) -> list[dict]:
    """Combine Yahoo + Google News, de-duplicated by headline.

    ``name`` (the company name) makes the Google query far more relevant than the
    bare ticker — pass ``Fundamentals.name`` when available.
    """
    query = f"{name} share price NSE" if name else f"{symbol} stock NSE"
    combined = fetch_google_news(query, limit=limit) + fetch_news(
        symbol, suffix=suffix, limit=limit)
    seen, out = set(), []
    for n in combined:
        key = (n.get("title") or "")[:50].lower()
        if key and key not in seen:
            seen.add(key)
            out.append(n)
    return out[:limit]
