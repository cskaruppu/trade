"""Local SQLite cache for precomputed scans.

Scanning thousands of NSE/BSE stocks is too slow to do on every dashboard
click. Instead a scheduled job (``nsetrade scan``, e.g. nightly via Windows
Task Scheduler) runs the heavy scan once and stores the ranked rows here; the
dashboard and CLI then read the latest result instantly. This is exactly how
the fast commercial screeners feel quick — precompute, don't compute-on-click.

Stored entirely on the user's machine (default ``~/.nsetrade/scans.db``); no
data leaves the laptop.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path.home() / ".nsetrade" / "scans.db"


class ScanCache:
    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    universe TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    meta TEXT NOT NULL,
                    rows TEXT NOT NULL
                )"""
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_scans_lookup "
                "ON scans (kind, universe, created_at)"
            )

    def save_run(self, kind: str, universe: str, rows: list[dict],
                 meta: Optional[dict] = None, *, now: Optional[float] = None) -> None:
        """Persist a completed scan. ``rows`` must be JSON-serialisable dicts."""
        ts = time.time() if now is None else now
        with self._conn() as c:
            c.execute(
                "INSERT INTO scans (kind, universe, created_at, meta, rows) "
                "VALUES (?, ?, ?, ?, ?)",
                (kind, universe, ts, json.dumps(meta or {}), json.dumps(rows)),
            )

    def latest(self, kind: str, universe: str) -> Optional[dict]:
        """Return the most recent run for ``(kind, universe)`` or ``None``.

        Result: ``{"created_at": float, "meta": dict, "rows": list[dict]}``.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT created_at, meta, rows FROM scans "
                "WHERE kind = ? AND universe = ? ORDER BY created_at DESC LIMIT 1",
                (kind, universe),
            ).fetchone()
        if not row:
            return None
        return {"created_at": row[0], "meta": json.loads(row[1]),
                "rows": json.loads(row[2])}

    def list_runs(self, limit: int = 20) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT kind, universe, created_at, "
                "json_array_length(rows) FROM scans "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"kind": r[0], "universe": r[1], "created_at": r[2],
                 "count": r[3]} for r in rows]

    def prune(self, keep_per_key: int = 5) -> int:
        """Keep only the newest ``keep_per_key`` runs per (kind, universe).

        Returns the number of rows deleted.
        """
        with self._conn() as c:
            keys = c.execute(
                "SELECT DISTINCT kind, universe FROM scans").fetchall()
            deleted = 0
            for kind, universe in keys:
                ids = [r[0] for r in c.execute(
                    "SELECT id FROM scans WHERE kind = ? AND universe = ? "
                    "ORDER BY created_at DESC", (kind, universe)).fetchall()]
                drop = ids[keep_per_key:]
                if drop:
                    c.executemany("DELETE FROM scans WHERE id = ?",
                                  [(i,) for i in drop])
                    deleted += len(drop)
        return deleted
