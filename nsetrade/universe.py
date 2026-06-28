"""NSE symbol universes for screening.

Symbols are bare NSE trading symbols (no exchange suffix); each provider adds
its own. Lists are static snapshots and may drift as indices rebalance — for a
production system, refresh these from the NSE indices CSVs or your broker's
instruments dump. ``custom`` can be supplied at the CLI with ``--symbols``.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

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


# --------------------------------------------------------------------------
# Full-exchange universes (NSE ~2000, BSE thousands) — loaded from a cached CSV
# --------------------------------------------------------------------------

# Where refreshed equity lists are cached on the user's machine.
CACHE_DIR = Path.home() / ".nsetrade"
# NSE publishes the canonical equity master here (free, no key).
NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

_FULL_FILES = {"nse_all": "nse_equity.csv", "bse_all": "bse_equity.csv"}


def parse_nse_equity_csv(text: str) -> list[str]:
    """Extract tradable symbols from NSE's EQUITY_L.csv (or a BSE list in the
    same SYMBOL/SERIES shape). Keeps the equity series (EQ/BE) only.

    NSE's header names are padded with spaces (e.g. ``" SERIES"``), so we match
    columns case- and space-insensitively.
    """
    reader = csv.DictReader(io.StringIO(text))
    out: list[str] = []
    for row in reader:
        norm = {(k or "").strip().upper(): (v or "").strip()
                for k, v in row.items()}
        sym = norm.get("SYMBOL", "")
        series = norm.get("SERIES", "")
        if sym and (series in ("EQ", "BE") or series == ""):
            out.append(sym.upper())
    return out


def refresh_nse_equity_list(dest: Path | None = None,
                            url: str = NSE_EQUITY_URL) -> list[str]:
    """Download NSE's equity master to the cache and return the symbol list.

    Network call — works on the user's laptop; needs outbound HTTPS to NSE.
    """
    import urllib.request

    dest = Path(dest) if dest else CACHE_DIR / _FULL_FILES["nse_all"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        text = resp.read().decode("utf-8", "replace")
    symbols = parse_nse_equity_csv(text)
    if not symbols:
        raise ValueError("downloaded NSE list parsed to 0 symbols")
    dest.write_text(text, encoding="utf-8")
    return symbols


def load_full_universe(name: str) -> list[str] | None:
    """Return a cached full-exchange list, or ``None`` if not cached yet."""
    fn = _FULL_FILES.get(name.lower())
    if not fn:
        return None
    path = CACHE_DIR / fn
    if not path.exists():
        return None
    return parse_nse_equity_csv(path.read_text(encoding="utf-8"))


def list_universes() -> list[str]:
    """All selectable universe names, including any cached full lists."""
    names = list(UNIVERSES)
    for full in _FULL_FILES:
        if (CACHE_DIR / _FULL_FILES[full]).exists():
            names.append(full)
    return names


def get_universe(name: str) -> list[str]:
    """Return the symbol list for a named universe.

    Named index snapshots (nifty50/100/500) come from the bundled lists; the
    full-exchange names (``nse_all``, ``bse_all``) load from the refreshed CSV
    cache (see :func:`refresh_nse_equity_list`).
    """
    key = (name or "nifty50").lower()
    if key in UNIVERSES:
        return list(UNIVERSES[key])
    full = load_full_universe(key)
    if full is not None:
        return full
    if key in _FULL_FILES:
        raise ValueError(
            f"universe {key!r} is not cached yet. Run "
            f"'nsetrade refresh-universe' first (downloads the NSE equity list)."
        )
    raise ValueError(
        f"unknown universe {name!r}. Available: {', '.join(list_universes())} "
        f"(or pass --symbols for a custom list)."
    )
