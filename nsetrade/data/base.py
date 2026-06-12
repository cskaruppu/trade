"""Abstract data-provider interface.

Every provider returns a pandas DataFrame indexed by a tz-naive DatetimeIndex
with exactly these columns (all floats except volume which is numeric):

    open, high, low, close, volume

This uniform contract is what lets indicators, patterns, the screener and the
backtester stay completely provider-agnostic. To add a new data source (your
broker, a CSV dump, a paid vendor) just subclass DataProvider and implement
``history``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    """Base class for all market-data providers."""

    #: human-readable provider name, e.g. "yfinance"
    name: str = "base"

    @abstractmethod
    def history(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        period_days: int = 400,
    ) -> pd.DataFrame:
        """Return OHLCV history for ``symbol``.

        Parameters
        ----------
        symbol:
            NSE trading symbol *without* any exchange suffix, e.g. ``RELIANCE``.
            Providers add their own suffix/exchange prefix internally.
        interval:
            Bar size. ``"1d"`` (daily) is the common case; providers may also
            support ``"1wk"``, ``"60minute"`` etc.
        period_days:
            How many calendar days of history to fetch, counting back from today.

        Returns
        -------
        pandas.DataFrame
            Indexed by date, columns == :data:`OHLCV_COLUMNS`, oldest row first.
        """
        raise NotImplementedError

    # ---- shared helpers -------------------------------------------------

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce an arbitrary OHLCV frame into the canonical contract."""
        df = df.rename(columns={c: c.lower() for c in df.columns})
        missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"data is missing required columns: {missing}")
        df = df[OHLCV_COLUMNS].copy()
        for col in OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
