"""Multi-timeframe confluence — when daily, weekly and monthly agree.

A setup that lines up across timeframes is higher-conviction than one that only
shows on the daily chart. This module scores a symbol on each timeframe and
combines them into a single conviction reading, weighting the higher timeframes
more for trend while still rewarding daily timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .data import get_provider
from .resample import resample_ohlcv, scale_period_days
from .signals.engine import Signal, signal_for_frame

# higher timeframes carry more weight (trend dominance), daily adds timing
_TF_WEIGHTS = {"daily": 1.0, "weekly": 1.6, "monthly": 2.0}


@dataclass
class Confluence:
    symbol: str
    per_timeframe: dict[str, Signal] = field(default_factory=dict)
    conviction: float = 0.0
    aligned: str = "mixed"          # "bullish" | "bearish" | "mixed"
    verdict: str = "Neutral"

    def as_row(self) -> dict:
        row = {"symbol": self.symbol, "conviction": round(self.conviction, 2),
               "aligned": self.aligned, "verdict": self.verdict}
        for tf, sig in self.per_timeframe.items():
            row[tf] = sig.verdict
        return row

    def describe(self) -> str:
        parts = [f"{tf}:{s.verdict}" for tf, s in self.per_timeframe.items()]
        return (f"{self.symbol}  conviction {self.conviction:+.2f} "
                f"[{self.aligned}]  " + "  ".join(parts))


def _verdict(conviction: float) -> str:
    if conviction >= 4:
        return "Strong Buy"
    if conviction >= 1.5:
        return "Buy"
    if conviction <= -4:
        return "Strong Sell"
    if conviction <= -1.5:
        return "Sell"
    return "Neutral"


def confluence_for_frames(symbol: str, frames: dict[str, pd.DataFrame]) -> Confluence:
    """Compute confluence from pre-fetched ``{timeframe: daily/weekly/monthly df}``."""
    per_tf: dict[str, Signal] = {}
    weighted = 0.0
    total_w = 0.0
    for tf, df in frames.items():
        if df is None or len(df) < 30:
            continue
        sig = signal_for_frame(symbol, df)
        per_tf[tf] = sig
        w = _TF_WEIGHTS.get(tf, 1.0)
        weighted += w * sig.score
        total_w += w

    conviction = weighted / total_w if total_w else 0.0

    # alignment: do all available timeframes point the same way?
    dirs = set()
    for sig in per_tf.values():
        if "Buy" in sig.verdict:
            dirs.add("bullish")
        elif "Sell" in sig.verdict:
            dirs.add("bearish")
        else:
            dirs.add("neutral")
    if dirs == {"bullish"}:
        aligned = "bullish"
        conviction += 1.0      # bonus for full agreement
    elif dirs == {"bearish"}:
        aligned = "bearish"
        conviction -= 1.0
    else:
        aligned = "mixed"

    return Confluence(symbol, per_tf, conviction, aligned, _verdict(conviction))


def confluence(
    symbol: str,
    *,
    provider: str = "yfinance",
    provider_config: Optional[dict] = None,
    timeframes: tuple[str, ...] = ("daily", "weekly", "monthly"),
    period_days: int = 400,
) -> Confluence:
    """Fetch daily data once, resample, and compute multi-timeframe confluence."""
    prov = get_provider(provider, provider_config)
    # fetch enough daily history for the highest timeframe requested
    max_days = max(scale_period_days(period_days, tf) for tf in timeframes)
    daily = prov.history(symbol, period_days=max_days)
    frames = {tf: resample_ohlcv(daily, tf) for tf in timeframes}
    return confluence_for_frames(symbol, frames)


def confluence_scan(
    symbols: list[str],
    *,
    provider: str = "yfinance",
    provider_config: Optional[dict] = None,
    timeframes: tuple[str, ...] = ("daily", "weekly", "monthly"),
    period_days: int = 400,
    on_progress=None,
) -> list[Confluence]:
    """Rank a list of symbols by multi-timeframe conviction (high to low)."""
    out: list[Confluence] = []
    for i, sym in enumerate(symbols):
        try:
            out.append(confluence(
                sym, provider=provider, provider_config=provider_config,
                timeframes=timeframes, period_days=period_days))
        except Exception:  # noqa: BLE001 - skip bad symbols
            pass
        if on_progress:
            on_progress(i + 1, len(symbols), sym)
    out.sort(key=lambda c: c.conviction, reverse=True)
    return out
