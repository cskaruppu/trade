"""Period breakout scanner — stocks making new N-period highs.

A core momentum screen used everywhere in the industry: is the stock breaking
out to a new high over the last 3 months / 6 months / 52 weeks / multiple years?
The 52-week-high breakout is the classic; shorter and longer windows catch
earlier and more significant moves respectively.

A "breakout" here means today's close has cleared the highest high of the prior
bars in the window (optionally within a small tolerance band, to catch stocks
right at the edge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

# trading-day lengths for the standard periods (None = all available history)
PERIODS = {
    "1 month": 21,
    "3 months": 63,
    "6 months": 126,
    "52 weeks": 252,
    "3 years": 756,
    "All-time": None,
}


@dataclass
class BreakoutInfo:
    symbol: str
    close: float
    period_high: float          # highest prior high in the window
    pct_from_high: float        # +ve = below the high, -ve = above it (new high)
    at_new_high: bool
    bars_since_high: int

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "close": round(self.close, 2),
            "period_high": round(self.period_high, 2),
            "vs high": f"{-self.pct_from_high:+.1%}",   # +ve = above prior high
            "new high": "✓" if self.at_new_high else "-",
            "bars since high": self.bars_since_high,
        }


def period_breakout(df: pd.DataFrame, *, lookback_days: Optional[int] = 252,
                    tol: float = 0.0, symbol: str = "") -> Optional[BreakoutInfo]:
    """Is the latest close making a new high over ``lookback_days`` bars?

    ``tol`` is a tolerance band (e.g. 0.02 = within 2% of the prior high also
    counts, for stocks right at the edge). Returns ``None`` if too little data.
    """
    if len(df) < 2:
        return None
    window = df.iloc[-lookback_days:] if lookback_days else df
    prior = window.iloc[:-1] if len(window) > 1 else window
    # use the prior CLOSING high — a robust "new high" definition that ignores
    # intraday wicks
    prior_high = float(prior["close"].max())
    close = float(df["close"].iloc[-1])
    if prior_high <= 0:
        return None
    pct_from_high = (prior_high - close) / prior_high
    at_new_high = close >= prior_high * (1 - tol)
    ph_pos = int(prior["close"].values.argmax())
    bars_since_high = (len(prior) - 1) - ph_pos
    return BreakoutInfo(symbol=symbol, close=close, period_high=prior_high,
                        pct_from_high=pct_from_high, at_new_high=at_new_high,
                        bars_since_high=bars_since_high)


def scan_breakouts(
    symbols: list[str],
    *,
    period: str = "52 weeks",
    tol: float = 0.0,
    provider: str = "yfinance",
    provider_config: Optional[dict] = None,
    period_days: int = 800,
    on_progress=None,
    _provider_obj=None,
):
    """Find stocks at a new ``period`` high. Returns ``(breakouts, errors)``.

    Sorted by how far they've cleared the prior high (strongest first).
    """
    from .data import get_provider

    lookback = PERIODS.get(period, 252)
    # fetch enough history to cover the window (plus headroom)
    need = max(period_days, (lookback or 252) + 60)
    prov = _provider_obj or get_provider(provider, provider_config)
    out: list[BreakoutInfo] = []
    errors: dict[str, str] = {}

    for i, sym in enumerate(symbols):
        try:
            df = prov.history(sym, period_days=need)
            info = period_breakout(df, lookback_days=lookback, tol=tol, symbol=sym)
            if info and info.at_new_high:
                out.append(info)
        except Exception as exc:  # noqa: BLE001
            errors[sym] = str(exc)
        if on_progress:
            on_progress(i + 1, len(symbols), sym)

    out.sort(key=lambda b: b.pct_from_high)   # most above the prior high first
    return out, errors
