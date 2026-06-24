"""Universe screener: run the signal engine across many symbols and rank them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..data import get_provider
from ..signals.engine import Signal, signal_for_frame


@dataclass
class ScreenResult:
    signals: list[Signal]
    errors: dict[str, str]

    def to_frame(self) -> pd.DataFrame:
        rows = [s.as_row() for s in self.signals]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
        return df

    def top(self, n: int = 10, bullish: bool = True) -> pd.DataFrame:
        df = self.to_frame()
        if df.empty:
            return df
        if bullish:
            return df.head(n)
        return df.tail(n).iloc[::-1].reset_index(drop=True)


def screen(
    symbols: list[str],
    *,
    provider: str = "yfinance",
    provider_config: Optional[dict] = None,
    period_days: int = 400,
    interval: str = "1d",
    timeframe: str = "daily",
    on_progress=None,
) -> ScreenResult:
    """Fetch each symbol, score it, and collect results.

    Network/data errors for individual symbols are captured in ``errors`` rather
    than aborting the whole scan. ``timeframe`` ("daily"/"weekly"/"monthly")
    resamples the daily data before scoring.
    """
    from ..resample import resample_ohlcv, scale_period_days

    prov = get_provider(provider, provider_config)
    fetch_days = scale_period_days(period_days, timeframe)
    signals: list[Signal] = []
    errors: dict[str, str] = {}

    for i, sym in enumerate(symbols):
        try:
            df = prov.history(sym, interval=interval, period_days=fetch_days)
            df = resample_ohlcv(df, timeframe)
            signals.append(signal_for_frame(sym, df))
        except Exception as exc:  # noqa: BLE001 - keep scanning on per-symbol error
            errors[sym] = str(exc)
        if on_progress:
            on_progress(i + 1, len(symbols), sym)

    return ScreenResult(signals=signals, errors=errors)
