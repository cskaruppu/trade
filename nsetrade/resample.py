"""Resample daily OHLCV bars to weekly or monthly timeframes.

Most data sources serve daily (and intraday) bars only, so the uniform way to
get weekly/monthly candles across every provider is to fetch daily data and
aggregate it here. Weekly bars are anchored to Friday (the NSE week close);
monthly bars to month-end.
"""

from __future__ import annotations

import pandas as pd

# pandas resample rules. "ME" = month-end (pandas >= 2.2); "W-FRI" = week ending
# Friday.
_RULES = {
    "daily": None,
    "weekly": "W-FRI",
    "monthly": "ME",
}

TIMEFRAMES = list(_RULES)

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

# Rough multiplier: how many extra daily bars to fetch so the resampled frame
# still has enough history for long indicators (e.g. 200-period MAs).
FETCH_MULTIPLIER = {"daily": 1, "weekly": 6, "monthly": 23}


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "daily") -> pd.DataFrame:
    """Aggregate a daily OHLCV frame to ``timeframe``.

    Parameters
    ----------
    df:
        Daily OHLCV frame (canonical columns: open, high, low, close, volume).
    timeframe:
        ``"daily"`` (returns ``df`` unchanged), ``"weekly"`` or ``"monthly"``.
    """
    tf = (timeframe or "daily").lower()
    if tf not in _RULES:
        raise ValueError(
            f"unknown timeframe {timeframe!r}. Use one of: {', '.join(TIMEFRAMES)}"
        )
    rule = _RULES[tf]
    if rule is None:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("resampling needs a DatetimeIndex")

    out = df.resample(rule).agg(_AGG)
    # drop incomplete buckets with no trades
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def scale_period_days(period_days: int, timeframe: str) -> int:
    """Scale a requested history window so a higher timeframe still has bars."""
    return period_days * FETCH_MULTIPLIER.get((timeframe or "daily").lower(), 1)
