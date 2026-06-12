"""Zerodha Kite Connect data provider.

Requires a paid Kite Connect subscription and the ``kiteconnect`` package
(``pip install -e ".[kite]"``). The ``access_token`` is generated daily through
the Kite login flow — see https://kite.trade/docs/connect/v3/.

Credentials are read, in priority order, from:
  1. explicit ``api_key`` / ``access_token`` arguments
  2. the ``providers.kite`` block in your ``config.yaml``
  3. the ``KITE_API_KEY`` / ``KITE_ACCESS_TOKEN`` environment variables
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import pandas as pd

from .base import DataProvider


class KiteProvider(DataProvider):
    name = "kite"

    # Kite uses these interval strings directly.
    _INTERVALS = {
        "1d": "day",
        "day": "day",
        "60minute": "60minute",
        "15minute": "15minute",
        "5minute": "5minute",
        "minute": "minute",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        exchange: str = "NSE",
    ):
        self.api_key = api_key or os.environ.get("KITE_API_KEY")
        self.access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN")
        self.exchange = exchange
        if not self.api_key or not self.access_token:
            raise ValueError(
                "Kite provider needs api_key and access_token. Set them in "
                "config.yaml (providers.kite) or via KITE_API_KEY / "
                "KITE_ACCESS_TOKEN environment variables."
            )
        try:
            from kiteconnect import KiteConnect
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "kiteconnect is not installed. Run: pip install -e \".[kite]\""
            ) from exc

        self._kite = KiteConnect(api_key=self.api_key)
        self._kite.set_access_token(self.access_token)
        self._instrument_cache: dict[str, int] = {}

    # ---- instrument lookup ---------------------------------------------

    def _instrument_token(self, symbol: str) -> int:
        """Resolve an NSE trading symbol to its Kite instrument token."""
        symbol = symbol.upper().strip()
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]
        # instruments() is a large download; cache the whole exchange once.
        if not self._instrument_cache:
            for inst in self._kite.instruments(self.exchange):
                self._instrument_cache[inst["tradingsymbol"]] = inst[
                    "instrument_token"
                ]
        if symbol not in self._instrument_cache:
            raise ValueError(
                f"symbol {symbol!r} not found on {self.exchange} via Kite"
            )
        return self._instrument_cache[symbol]

    def history(
        self,
        symbol: str,
        *,
        interval: str = "1d",
        period_days: int = 400,
    ) -> pd.DataFrame:
        kite_interval = self._INTERVALS.get(interval, interval)
        token = self._instrument_token(symbol)
        to_date = dt.date.today()
        from_date = to_date - dt.timedelta(days=period_days)
        records = self._kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=kite_interval,
        )
        if not records:
            raise ValueError(f"no data returned for {symbol!r} from Kite")
        df = pd.DataFrame.from_records(records)
        df = df.set_index("date")
        return self._normalise(df)
