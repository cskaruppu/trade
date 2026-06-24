"""Personal watchlist management with NSE CSV import.

Your watchlist is stored as a simple text file (one symbol per line) at
``watchlist.txt`` in the project root by default. It is git-ignored so your
picks stay private. You can also import the CSVs that NSE lets you download
(equity master list, index constituents, or a "market watch" export) — the
importer finds the symbol column automatically.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Optional

DEFAULT_PATH = "watchlist.txt"

# Column names NSE uses for the trading symbol across its various CSV exports.
_SYMBOL_HEADERS = ("symbol", "symbol ", "tradingsymbol", "ticker", "scrip",
                   "security")


def _clean(sym: str) -> str:
    return sym.strip().upper().replace(" ", "")


def load(path: str = DEFAULT_PATH) -> list[str]:
    """Return the saved watchlist (empty if the file does not exist)."""
    if not os.path.exists(path):
        return []
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(_clean(line))
    # de-dupe, preserve order
    seen: set[str] = set()
    return [s for s in out if not (s in seen or seen.add(s))]


def save(symbols: Iterable[str], path: str = DEFAULT_PATH) -> list[str]:
    """Persist ``symbols`` (de-duped, order-preserving) and return the list."""
    seen: set[str] = set()
    cleaned = []
    for s in symbols:
        c = _clean(s)
        if c and c not in seen:
            seen.add(c)
            cleaned.append(c)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# nsetrade watchlist — one NSE symbol per line\n")
        for s in cleaned:
            fh.write(s + "\n")
    return cleaned


def add(symbols: Iterable[str], path: str = DEFAULT_PATH) -> list[str]:
    """Add one or more symbols to the watchlist."""
    current = load(path)
    return save(current + [_clean(s) for s in symbols], path)


def remove(symbols: Iterable[str], path: str = DEFAULT_PATH) -> list[str]:
    """Remove one or more symbols from the watchlist."""
    drop = {_clean(s) for s in symbols}
    return save([s for s in load(path) if s not in drop], path)


def clear(path: str = DEFAULT_PATH) -> None:
    """Empty the watchlist."""
    save([], path)


def _detect_symbol_column(header: list[str]) -> Optional[int]:
    lowered = [h.strip().lower() for h in header]
    for i, h in enumerate(lowered):
        if h in _SYMBOL_HEADERS:
            return i
    # fall back to any column containing 'symbol'
    for i, h in enumerate(lowered):
        if "symbol" in h:
            return i
    return None


def import_csv(
    csv_path: str,
    *,
    column: Optional[str] = None,
    merge: bool = True,
    path: str = DEFAULT_PATH,
) -> list[str]:
    """Import symbols from an NSE CSV export into the watchlist.

    Parameters
    ----------
    csv_path:
        Path to the downloaded CSV (e.g. NSE's ``EQUITY_L.csv``, an index
        constituents file, or a market-watch export).
    column:
        Force a specific column name to read symbols from. If omitted, the
        importer auto-detects a ``SYMBOL``-like column.
    merge:
        If True (default) add to the existing watchlist; if False replace it.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r]
    if not rows:
        raise ValueError(f"{csv_path} is empty")

    header = rows[0]
    if column is not None:
        lowered = [h.strip().lower() for h in header]
        try:
            col_idx = lowered.index(column.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"column {column!r} not found. Columns: {header}"
            ) from exc
        data_rows = rows[1:]
    else:
        col_idx = _detect_symbol_column(header)
        if col_idx is None:
            # no header match — assume single-column file with no header
            if len(header) == 1:
                col_idx = 0
                data_rows = rows  # treat all lines as data
            else:
                raise ValueError(
                    "could not find a symbol column; pass column=... "
                    f"explicitly. Columns seen: {header}"
                )
        else:
            data_rows = rows[1:]

    symbols = []
    for r in data_rows:
        if col_idx < len(r):
            val = _clean(r[col_idx])
            if val and val.isascii():
                symbols.append(val)

    if not symbols:
        raise ValueError(f"no symbols parsed from {csv_path}")

    if merge:
        return add(symbols, path)
    return save(symbols, path)
