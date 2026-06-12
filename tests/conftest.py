"""Shared synthetic-data fixtures so tests never touch the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(prices: np.ndarray, start="2021-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(prices))
    close = pd.Series(prices, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.005
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.995
    vol = pd.Series(np.random.RandomState(0).randint(1e5, 1e6, len(prices)),
                    index=idx)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


@pytest.fixture
def trending_df():
    """An uptrend then downtrend — long enough for 200-SMA based signals."""
    n = 320
    up = np.linspace(100, 200, n // 2)
    down = np.linspace(200, 140, n - n // 2)
    prices = np.concatenate([up, down])
    # add mild noise
    prices = prices + np.sin(np.arange(len(prices)) / 5) * 2
    return _make_ohlcv(prices)


@pytest.fixture
def make_ohlcv():
    return _make_ohlcv
