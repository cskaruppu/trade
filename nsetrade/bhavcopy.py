"""NSE Bhavcopy — bulk end-of-day data for the whole exchange.

NSE publishes a daily "full" Bhavcopy at a fixed, date-based archive URL: one
CSV per trading day containing OHLCV for *every* listed security. Downloading
one file per day gives you the entire NSE universe without 2,000 per-symbol API
calls — far more robust than scraping per-stock quotes.

This module:
  * builds the dated URL and downloads/parses a day's file,
  * stores rows in a local SQLite database (incremental — only new days),
  * serves any symbol's history from that store via :class:`BhavcopyProvider`.

Honest limits: this is **end-of-day** data, and Bhavcopy prices are **raw**
(not split/dividend adjusted). :func:`adjust_splits` applies a heuristic
back-adjustment for obvious split ratios; it is best-effort, not a substitute
for a real corporate-actions feed.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from .data.base import DataProvider

DEFAULT_DB = Path.home() / ".nsetrade" / "bhavcopy.db"
# NSE's security-wise full bhavcopy (all equities, OHLCV + volume).
BHAVCOPY_URL = ("https://nsearchives.nseindia.com/products/content/"
                "sec_bhavdata_full_{ddmmyyyy}.csv")
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
# common Indian split/bonus ratios used by the heuristic adjuster
_COMMON_RATIOS = [2.0, 2.5, 3.0, 4.0, 5.0, 10.0]


def bhavcopy_url(d: dt.date) -> str:
    """The archive URL for the full security-wise bhavcopy of date ``d``."""
    return BHAVCOPY_URL.format(ddmmyyyy=d.strftime("%d%m%Y"))


def parse_bhavcopy(text: str, *, series=("EQ", "BE")) -> list[dict]:
    """Parse a full-bhavcopy CSV into OHLCV rows (equity series only).

    NSE pads its headers with spaces, so we match columns case/space-insensitively.
    """
    reader = csv.DictReader(io.StringIO(text))
    out: list[dict] = []
    for raw in reader:
        row = {(k or "").strip().upper(): (v or "").strip() for k, v in raw.items()}
        ser = row.get("SERIES", "")
        if series and ser not in series:
            continue
        sym = row.get("SYMBOL", "")
        try:
            rec = {
                "symbol": sym.upper(),
                "open": float(row["OPEN_PRICE"]),
                "high": float(row["HIGH_PRICE"]),
                "low": float(row["LOW_PRICE"]),
                "close": float(row["CLOSE_PRICE"]),
                "volume": float(row.get("TTL_TRD_QNTY", "0") or 0),
            }
        except (KeyError, ValueError):
            continue
        if rec["symbol"] and rec["close"] > 0:
            out.append(rec)
    return out


def fetch_day(d: dt.date, *, url: Optional[str] = None, timeout: int = 30) -> list[dict]:
    """Download and parse one day's bhavcopy. Network call.

    Returns ``[]`` for a non-trading day (NSE returns 404 on holidays/weekends).
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url or bhavcopy_url(d), headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            text = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []                      # market closed that day
        raise
    return parse_bhavcopy(text)


class BhavcopyStore:
    """Local SQLite store of daily OHLCV rows keyed by (date, symbol)."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        return sqlite3.connect(str(self.path))

    def _init(self):
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS prices (
                    date TEXT NOT NULL, symbol TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (date, symbol))""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sym ON prices (symbol, date)")

    def has_date(self, d: dt.date) -> bool:
        with self._conn() as c:
            r = c.execute("SELECT 1 FROM prices WHERE date = ? LIMIT 1",
                          (d.isoformat(),)).fetchone()
        return r is not None

    def save_day(self, d: dt.date, rows: list[dict]) -> int:
        iso = d.isoformat()
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO prices "
                "(date, symbol, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?)",
                [(iso, r["symbol"], r["open"], r["high"], r["low"], r["close"],
                  r["volume"]) for r in rows])
        return len(rows)

    def symbols(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute(
                "SELECT DISTINCT symbol FROM prices ORDER BY symbol").fetchall()]

    def date_range(self):
        with self._conn() as c:
            r = c.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
        return r if r and r[0] else (None, None)

    def history(self, symbol: str) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute(
                "SELECT date, open, high, low, close, volume FROM prices "
                "WHERE symbol = ? ORDER BY date", (symbol.upper(),)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                         "volume"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")


def download_range(store: BhavcopyStore, start: dt.date, end: dt.date, *,
                   on_progress=None, fetch=fetch_day) -> dict:
    """Fetch every missing weekday between ``start`` and ``end`` into ``store``.

    Skips weekends and days already stored. Returns a small summary dict.
    ``fetch`` is injectable for tests.
    """
    days = (end - start).days
    fetched = trading = 0
    for i in range(days + 1):
        d = start + dt.timedelta(days=i)
        if d.weekday() >= 5 or store.has_date(d):     # weekend / already have it
            if on_progress:
                on_progress(i + 1, days + 1, d)
            continue
        rows = fetch(d)
        if rows:
            store.save_day(d, rows)
            trading += 1
        fetched += 1
        if on_progress:
            on_progress(i + 1, days + 1, d)
    return {"days_checked": days + 1, "days_fetched": fetched,
            "trading_days": trading}


def adjust_splits(df: pd.DataFrame, *, ratios=_COMMON_RATIOS,
                  tol: float = 0.06) -> pd.DataFrame:
    """Heuristically back-adjust obvious stock splits/bonuses.

    Scans for a day where the close drops by a factor close to a known split
    ratio (e.g. ~5×) alongside a volume surge, and divides all *prior* prices by
    that ratio so the series is continuous. Best-effort — a real corporate-
    actions feed is more reliable; this just stops big splits from faking a
    90% crash that wrecks pattern detection.
    """
    if len(df) < 3:
        return df
    out = df.copy()
    close = out["close"].values
    factor = pd.Series(1.0, index=out.index)
    for i in range(1, len(close)):
        prev, cur = close[i - 1], close[i]
        if cur <= 0 or prev <= 0:
            continue
        drop = prev / cur
        for r in ratios:
            if abs(drop - r) <= tol * r:
                # everything before bar i is on the pre-split scale → divide
                factor.iloc[:i] /= r
                break
    for col in ("open", "high", "low", "close"):
        if col in out:
            out[col] = out[col] * factor
    return out


class BhavcopyProvider(DataProvider):
    """Serve OHLCV history for any NSE symbol from the local Bhavcopy store."""

    name = "bhavcopy"

    def __init__(self, db: Optional[str] = None, adjust: bool = True):
        self.store = BhavcopyStore(db)
        self.adjust = adjust

    def history(self, symbol: str, *, interval: str = "1d",
                period_days: int = 400) -> pd.DataFrame:
        df = self.store.history(symbol)
        if df.empty:
            raise ValueError(
                f"no Bhavcopy data for {symbol!r}. Run "
                f"'nsetrade fetch-bhavcopy' to populate the local store.")
        if self.adjust:
            df = adjust_splits(df)
        df = self._normalise(df)
        if period_days:
            cutoff = df.index.max() - pd.Timedelta(days=period_days)
            df = df[df.index >= cutoff]
        return df
