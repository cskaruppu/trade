"""Focused pattern screener — find stocks forming specific chart patterns.

Scans a universe for a chosen set of structural patterns (e.g. Cup & Handle,
Darvas Box) and attaches each pattern's *validated historical edge on that
stock* — so the result isn't just "this looks like a cup", it's "this is a cup,
and cups have paid off here N times, and the edge held out-of-sample".

Sorted so the most trustworthy setups (robust edge, high win-rate, confirmed
breakout) rise to the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# the two the user asked to track, by detector key
DEFAULT_PATTERNS = ["cup_and_handle", "darvas_box"]


@dataclass
class PatternHit:
    symbol: str
    pattern: str
    direction: str
    status: str                 # "forming" | "breakout"
    breakout_level: Optional[float]
    close: float
    edge_win_rate: Optional[float] = None
    edge_occurrences: Optional[int] = None
    edge_robust: Optional[bool] = None
    volume_confirmed: Optional[bool] = None
    target: Optional[float] = None        # measured-move target
    upside_pct: Optional[float] = None     # implied % move from current close
    confirm: Optional[str] = None          # breakout_check state
    note: str = ""

    _SIGNAL = {"confirmed": "✅ confirmed", "volume_light": "⚠️ weak vol",
               "approaching": "⏳ near", "far": "🔶 building"}

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "pattern": self.pattern,
            "signal": self._SIGNAL.get(self.confirm or "", "-"),
            "status": self.status,
            "vol": ("✓" if self.volume_confirmed else
                    ("✗" if self.volume_confirmed is False else "-")),
            "close": round(self.close, 2),
            "breakout": round(self.breakout_level, 2) if self.breakout_level else None,
            "target": round(self.target, 2) if self.target else None,
            "upside %": round(self.upside_pct * 100, 1)
            if self.upside_pct is not None else None,
            "edge": (f"{self.edge_win_rate:.0%} / {self.edge_occurrences}"
                     if self.edge_win_rate is not None else "-"),
            "robust": ("✓" if self.edge_robust else
                       ("✗" if self.edge_robust is False else "-")),
            "note": self.note,
        }


def scan_for_patterns(
    symbols: list[str],
    pattern_keys: Optional[list[str]] = None,
    *,
    provider: str = "yfinance",
    provider_config: Optional[dict] = None,
    period_days: int = 500,
    with_edge: bool = True,
    only_breakouts: bool = False,
    only_volume_confirmed: bool = False,
    only_confirmed: bool = False,
    on_progress=None,
    _provider_obj=None,
):
    """Scan ``symbols`` for the requested patterns. Returns ``(hits, errors)``.

    ``only_volume_confirmed`` keeps only breakouts backed by above-average
    volume; ``only_confirmed`` keeps only *confirmed* breakouts (a close above
    the trigger on above-average volume). Pass ``_provider_obj`` to inject a
    provider in tests.
    """
    from .ai import _pattern_key
    from .data import get_provider
    from .explain import breakout_check
    from .patterns import detect_advanced

    keys = set(pattern_keys or DEFAULT_PATTERNS)
    prov = _provider_obj or get_provider(provider, provider_config)
    hits: list[PatternHit] = []
    errors: dict[str, str] = {}

    for i, sym in enumerate(symbols):
        try:
            df = prov.history(sym, period_days=period_days)
            close = float(df["close"].iloc[-1])
            for m in detect_advanced(df):
                key = _pattern_key(m.name)
                if key not in keys:
                    continue
                if only_breakouts and m.status != "breakout":
                    continue
                if only_volume_confirmed and not m.volume_confirmed:
                    continue
                _bc = breakout_check(df, m)
                if only_confirmed and _bc.state != "confirmed":
                    continue
                # measured-move target (breakout + the base's own height) and
                # the % upside it implies from the current price
                target = upside = None
                if (m.breakout_level and m.support
                        and m.breakout_level > m.support and close > 0):
                    target = m.breakout_level + (m.breakout_level - m.support)
                    upside = target / close - 1.0
                hit = PatternHit(symbol=sym, pattern=m.name, direction=m.direction,
                                 status=m.status, breakout_level=m.breakout_level,
                                 close=close, volume_confirmed=m.volume_confirmed,
                                 target=target, upside_pct=upside,
                                 confirm=_bc.state, note=m.note)
                if with_edge:
                    try:
                        from .edge import pattern_edge_validated
                        ve = pattern_edge_validated(df, key)
                        if ve.full.occurrences:
                            hit.edge_win_rate = ve.full.win_rate
                            hit.edge_occurrences = ve.full.occurrences
                            hit.edge_robust = ve.robust
                    except Exception:  # noqa: BLE001 - edge is best-effort
                        pass
                hits.append(hit)
        except Exception as exc:  # noqa: BLE001 - keep scanning
            errors[sym] = str(exc)
        if on_progress:
            on_progress(i + 1, len(symbols), sym)

    # best first: confirmed breakout → robust edge → volume → win-rate → breakout
    hits.sort(key=lambda h: (h.confirm == "confirmed",
                             h.edge_robust is True,
                             h.volume_confirmed is True,
                             h.edge_win_rate if h.edge_win_rate is not None else -1,
                             h.status == "breakout"), reverse=True)
    return hits, errors
