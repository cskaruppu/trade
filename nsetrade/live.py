"""Live / intraday watchlist scanning.

Polls a watchlist on an interval, recomputes the signal for each symbol, and
emits an *alert* whenever a symbol's verdict crosses into Buy/Sell territory or
a fresh pattern fires on the latest bar. Works with any provider but is intended
for your Zerodha Kite feed during market hours.

The scan logic (:func:`scan_once`, :func:`diff_alerts`) is pure and unit-tested;
:func:`watch` is the thin polling loop around it.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .data import get_provider
from .signals.engine import Signal, signal_for_frame

# NSE regular session in IST.
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)

_ACTIONABLE = {"Strong Buy", "Buy", "Sell", "Strong Sell"}


@dataclass
class Alert:
    symbol: str
    verdict: str
    score: float
    close: float
    reasons: list[str]
    kind: str  # "verdict_change" | "new_pattern"

    def line(self) -> str:
        arrow = "🟢" if "Buy" in self.verdict else "🔴" if "Sell" in self.verdict else "⚪"
        why = ", ".join(self.reasons[:3]) if self.reasons else "—"
        return (f"{arrow} {self.symbol:<12} {self.verdict:<11} "
                f"score {self.score:+.1f}  @ {self.close:.2f}  [{self.kind}]  {why}")


def is_market_open(now: Optional[dt.datetime] = None) -> bool:
    """True if the NSE regular session is open at ``now`` (defaults to now IST)."""
    now = now or dt.datetime.now(IST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    now = now.astimezone(IST)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def scan_once(
    symbols: list[str],
    *,
    provider: str = "kite",
    provider_config: Optional[dict] = None,
    interval: str = "1d",
    period_days: int = 400,
) -> dict[str, Signal]:
    """Fetch + score every symbol once. Bad symbols are skipped silently."""
    prov = get_provider(provider, provider_config)
    out: dict[str, Signal] = {}
    for sym in symbols:
        try:
            df = prov.history(sym, interval=interval, period_days=period_days)
            out[sym] = signal_for_frame(sym, df)
        except Exception:  # noqa: BLE001 - keep scanning the rest of the list
            continue
    return out


def diff_alerts(
    prev: dict[str, Signal],
    curr: dict[str, Signal],
) -> list[Alert]:
    """Return alerts for verdict changes or newly-fired patterns vs ``prev``."""
    alerts: list[Alert] = []
    for sym, sig in curr.items():
        old = prev.get(sym)
        # verdict crossing into/within actionable territory
        if sig.verdict in _ACTIONABLE and (old is None or old.verdict != sig.verdict):
            alerts.append(Alert(sym, sig.verdict, sig.score, sig.close,
                                sig.reasons, "verdict_change"))
            continue
        # new reasons appearing while already actionable
        if old is not None and sig.verdict in _ACTIONABLE:
            new_reasons = [r for r in sig.reasons if r not in set(old.reasons)]
            if new_reasons:
                alerts.append(Alert(sym, sig.verdict, sig.score, sig.close,
                                    new_reasons, "new_pattern"))
    return alerts


def watch(
    symbols: list[str],
    *,
    provider: str = "kite",
    provider_config: Optional[dict] = None,
    interval_seconds: int = 300,
    poll_interval: str = "1d",
    only_market_hours: bool = True,
    on_alert: Optional[Callable[[Alert], None]] = None,
    max_iterations: Optional[int] = None,
) -> None:
    """Poll ``symbols`` every ``interval_seconds`` and dispatch alerts.

    Runs until interrupted (Ctrl-C) or ``max_iterations`` is reached. By default
    it only scans during NSE market hours.
    """
    emit = on_alert or (lambda a: print(a.line()))
    prev: dict[str, Signal] = {}
    iterations = 0

    print(f"watching {len(symbols)} symbols via {provider} "
          f"every {interval_seconds}s (Ctrl-C to stop)")
    try:
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            if only_market_hours and not is_market_open():
                print(f"[{dt.datetime.now(IST):%H:%M}] market closed — sleeping")
            else:
                curr = scan_once(symbols, provider=provider,
                                 provider_config=provider_config,
                                 interval=poll_interval)
                for alert in diff_alerts(prev, curr):
                    emit(alert)
                prev = curr
            if max_iterations is None or iterations < max_iterations:
                time.sleep(interval_seconds)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nstopped watching")
