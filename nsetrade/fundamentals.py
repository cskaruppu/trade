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
        return [n for n in out if n.get("title")]
    except Exception:  # noqa: BLE001
        return []
