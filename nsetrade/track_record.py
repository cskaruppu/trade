"""Honest signal track record — log signals, measure outcomes, score them.

The single biggest trust-builder for a trading tool is proof it works *over
time*, accounting for losses as well as wins. This module:

  1. **logs** each signal the system fires (symbol, pattern, entry, target, stop,
     confidence, date) into a local SQLite store,
  2. **evaluates** matured signals against what price actually did — did it hit
     the target first, the stop first, or neither within the window? — and
  3. **scores** the resolved set into an honest report card: win rate,
     hit-target / hit-stop rate, average return, profit factor, expectancy,
     broken down by pattern or confidence.

The outcome math is deliberately conservative (if a bar touches both target and
stop, the stop is assumed hit first) so the scorecard never flatters itself.
Everything except the network fetch is pure and unit-tested.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DB = Path.home() / ".nsetrade" / "track_record.db"


@dataclass
class TrackedSignal:
    id: int
    symbol: str
    pattern: str
    side: str            # "long" | "short"
    confidence: str
    entry: float
    target: float
    stop: float
    entry_date: str      # ISO date
    status: str          # "open" | "resolved"
    outcome: Optional[str] = None       # "target" | "stop" | "timeout"
    ret: Optional[float] = None         # signed return of the trade
    resolved_date: Optional[str] = None


def _signed_return(side: str, entry: float, exit_px: float) -> float:
    if entry <= 0:
        return 0.0
    return (exit_px / entry - 1.0) if side == "long" else (entry / exit_px - 1.0)


def resolve_signal(sig: TrackedSignal, forward: pd.DataFrame,
                   forward_bars: int) -> Optional[tuple]:
    """Decide a signal's outcome from the bars *after* its entry date.

    ``forward`` is the price history strictly after ``entry_date`` (oldest
    first). Returns ``(outcome, signed_return, resolved_date)`` once decided, or
    ``None`` if not enough bars have elapsed and neither level was hit yet.
    Conservative: a bar touching both target and stop counts as a stop.
    """
    window = forward.iloc[:forward_bars]
    for i in range(len(window)):
        hi = float(window["high"].iloc[i])
        lo = float(window["low"].iloc[i])
        when = window.index[i]
        if sig.side == "long":
            hit_stop = lo <= sig.stop
            hit_target = hi >= sig.target
        else:
            hit_stop = hi >= sig.stop
            hit_target = lo <= sig.target
        if hit_stop:
            return "stop", _signed_return(sig.side, sig.entry, sig.stop), str(when.date())
        if hit_target:
            return "target", _signed_return(sig.side, sig.entry, sig.target), str(when.date())
    if len(window) >= forward_bars:
        last = window.iloc[-1]
        ret = _signed_return(sig.side, sig.entry, float(last["close"]))
        return "timeout", ret, str(window.index[-1].date())
    return None      # still open — window not full and no level hit


def scorecard(signals: list[TrackedSignal]) -> dict:
    """Aggregate resolved signals into honest performance stats."""
    res = [s for s in signals if s.status == "resolved" and s.ret is not None]
    n = len(res)
    if n == 0:
        return {"n": 0}
    rets = [s.ret for s in res]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "hit_target_rate": sum(1 for s in res if s.outcome == "target") / n,
        "hit_stop_rate": sum(1 for s in res if s.outcome == "stop") / n,
        "avg_return": sum(rets) / n,
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0
        else (float("inf") if gross_win > 0 else 0.0),
        "expectancy": sum(rets) / n,
    }


def equity_curve(signals: list[TrackedSignal]) -> list[dict]:
    """Equal-weight, sequential equity curve from resolved signals.

    Orders resolved signals by resolution date and compounds their returns as
    if each were one unit of capital taken in turn. Returns points with running
    ``equity`` (starts at 1.0) and ``drawdown`` (from the running peak). A
    visualization aid — it assumes non-overlapping equal-size trades.
    """
    res = sorted(
        [s for s in signals if s.status == "resolved" and s.ret is not None
         and s.resolved_date],
        key=lambda s: s.resolved_date)
    eq, peak, pts = 1.0, 1.0, []
    for s in res:
        eq *= (1.0 + s.ret)
        peak = max(peak, eq)
        pts.append({"date": s.resolved_date, "symbol": s.symbol, "ret": s.ret,
                    "equity": eq, "drawdown": eq / peak - 1.0})
    return pts


def max_drawdown(curve: list[dict]) -> float:
    """Worst peak-to-trough drawdown across an equity curve (<= 0)."""
    return min((p["drawdown"] for p in curve), default=0.0)


class TrackRecord:
    def __init__(self, db: Optional[str | Path] = None):
        self.path = Path(db) if db else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        return sqlite3.connect(str(self.path))

    def _init(self):
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, pattern TEXT, side TEXT, confidence TEXT,
                    entry REAL, target REAL, stop REAL, entry_date TEXT,
                    status TEXT DEFAULT 'open', outcome TEXT, ret REAL,
                    resolved_date TEXT, created_at REAL)""")

    def log_signal(self, symbol, pattern, side, entry, target, stop, *,
                   confidence="", entry_date=None, now=None) -> bool:
        """Log a new signal. Skips if an OPEN one for the same
        symbol+pattern+side already exists (avoids re-logging daily). Returns
        True if a row was inserted."""
        entry_date = entry_date or pd.Timestamp(
            time.time() if now is None else now, unit="s").strftime("%Y-%m-%d")
        with self._conn() as c:
            dup = c.execute(
                "SELECT 1 FROM signals WHERE symbol=? AND pattern=? AND side=? "
                "AND status='open' LIMIT 1", (symbol, pattern, side)).fetchone()
            if dup:
                return False
            c.execute(
                "INSERT INTO signals (symbol,pattern,side,confidence,entry,target,"
                "stop,entry_date,status,created_at) VALUES (?,?,?,?,?,?,?,?,'open',?)",
                (symbol, pattern, side, confidence, float(entry), float(target),
                 float(stop), entry_date, time.time() if now is None else now))
        return True

    def _rows(self, where="", args=()) -> list[TrackedSignal]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id,symbol,pattern,side,confidence,entry,target,stop,"
                "entry_date,status,outcome,ret,resolved_date FROM signals "
                + where, args).fetchall()
        return [TrackedSignal(*r) for r in rows]

    def open_signals(self) -> list[TrackedSignal]:
        return self._rows("WHERE status='open'")

    def all_signals(self) -> list[TrackedSignal]:
        return self._rows("ORDER BY created_at DESC")

    def _mark_resolved(self, sig_id, outcome, ret, rdate):
        with self._conn() as c:
            c.execute("UPDATE signals SET status='resolved', outcome=?, ret=?, "
                      "resolved_date=? WHERE id=?", (outcome, ret, rdate, sig_id))

    def evaluate(self, *, fetch, forward_bars: int = 20, on_progress=None) -> int:
        """Resolve any matured open signals. ``fetch(symbol)`` returns the full
        OHLCV history (date-indexed). Returns the number newly resolved."""
        open_sigs = self.open_signals()
        resolved = 0
        for i, sig in enumerate(open_sigs):
            try:
                df = fetch(sig.symbol)
                entry_ts = pd.Timestamp(sig.entry_date)
                fwd = df[df.index > entry_ts]
                outcome = resolve_signal(sig, fwd, forward_bars)
                if outcome:
                    self._mark_resolved(sig.id, *outcome)
                    resolved += 1
            except Exception:  # noqa: BLE001 - keep going on per-symbol errors
                pass
            if on_progress:
                on_progress(i + 1, len(open_sigs), sig.symbol)
        return resolved

    def equity_curve(self) -> list[dict]:
        return equity_curve(self.all_signals())

    def scorecard(self, *, by: Optional[str] = None) -> dict:
        """Overall scorecard, or a dict of group → scorecard when ``by`` is
        'pattern' or 'confidence'."""
        sigs = self.all_signals()
        if by is None:
            return scorecard(sigs)
        groups: dict[str, list] = {}
        for s in sigs:
            key = getattr(s, by, "") or "—"
            groups.setdefault(key, []).append(s)
        return {k: scorecard(v) for k, v in groups.items()}
