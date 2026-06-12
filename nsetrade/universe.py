"""NSE symbol universes for screening.

Symbols are bare NSE trading symbols (no exchange suffix); each provider adds
its own. Lists are static snapshots and may drift as indices rebalance — for a
production system, refresh these from the NSE indices CSVs or your broker's
instruments dump. ``custom`` can be supplied at the CLI with ``--symbols``.
"""

from __future__ import annotations

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "ULTRACEMCO", "WIPRO", "JIOFIN", "ADANIGREEN",
]

# A handful of additional large/mid caps to extend coverage beyond Nifty 50.
_NIFTY_NEXT = [
    "DMART", "PIDILITIND", "GODREJCP", "DABUR", "MARICO",
    "HAVELLS", "SIEMENS", "AMBUJACEM", "BANKBARODA", "PNB",
    "GAIL", "IOC", "BPCL", "VEDL", "DLF",
    "ZOMATO", "PAYTM", "NAUKRI", "LICI", "IRCTC",
    "TVSMOTOR", "MOTHERSON", "BOSCHLTD", "INDIGO", "CHOLAFIN",
    "ICICIGI", "ICICIPRULI", "SBICARD", "MUTHOOTFIN", "BAJAJHLDNG",
]

NIFTY_100 = NIFTY_50 + _NIFTY_NEXT

UNIVERSES = {
    "nifty50": NIFTY_50,
    "nifty100": NIFTY_100,
    # alias kept for convenience; same as nifty100 snapshot here
    "nifty500": NIFTY_100,
}


def get_universe(name: str) -> list[str]:
    """Return the symbol list for a named universe."""
    key = (name or "nifty50").lower()
    if key not in UNIVERSES:
        raise ValueError(
            f"unknown universe {name!r}. Available: {', '.join(UNIVERSES)} "
            f"(or pass --symbols for a custom list)."
        )
    return list(UNIVERSES[key])
