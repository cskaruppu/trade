"""Free data provider backed by Yahoo Finance (the ``yfinance`` package).

NSE symbols on Yahoo carry a ``.NS`` suffix (e.g. ``RELIANCE.NS``); this
provider adds it for you. Install with ``pip install -e ".[yfinance]"``.
"""

from __future__ import annotations

import pandas as pd

from .base import DataProvider


class YFinanceProvider(DataProvider):
    name = "yfinance"

    # map our interval names to yfinance's
    _INTERVALS = {
        "1d": "1d",
        "1wk": "1wk",
        "1mo": "1mo",
        "60minute": "60m",
        "15minute": "15m",
        "5minute": "5m",
    }

    def __init__(self, suffix: str = ".NS"):
        self.suffix = suffix
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "yfinance is not installed. Run: pip install -e \".[yfinance]\""
            ) from exc

    def _yf_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().strip()
        if symbol.endswith(self.suffix) or "." in symbol:
            return symbol
        return f"{symbol}{self.suffix}"

    def history(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        period_days: int = 400,
    ) -> pd.DataFrame:
        import yfinance as yf

        yf_interval = self._INTERVALS.get(interval, interval)
        ticker = yf.Ticker(self._yf_symbol(symbol))
        # period in days; pad a little so we always have enough bars
        raw = ticker.history(
            period=f"{max(period_days, 5)}d",
            interval=yf_interval,
            auto_adjust=False,
        )
        if raw is None or raw.empty:
            raise ValueError(
                f"no data returned for {symbol!r} (yfinance symbol "
                f"{self._yf_symbol(symbol)!r}). Check the symbol is valid."
            )
        raw = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return self._normalise(raw)
