"""Regression test: _normalise must coerce pandas nullable dtypes to numpy float64.

Newer yfinance/pandas can return OHLCV as nullable extension dtypes (Float64 /
Int64). Their `.values` is an ExtensionArray, which made the pattern detectors
raise "boolean value of NA is ambiguous" on the daily (un-aggregated) frame.
"""

import numpy as np
import pandas as pd

from nsetrade.data.base import DataProvider
from nsetrade.patterns import detect_advanced
from nsetrade.signals.engine import signal_for_frame


def test_normalise_coerces_nullable_to_numpy_float64():
    n = 220
    c = np.linspace(80, 200, n) + np.sin(np.arange(n) / 15) * 8
    df = pd.DataFrame({
        "open": pd.array(c, dtype="Float64"),
        "high": pd.array(c * 1.01, dtype="Float64"),
        "low": pd.array(c * 0.99, dtype="Float64"),
        "close": pd.array(c, dtype="Float64"),
        "volume": pd.array([100000] * n, dtype="Int64"),
    }, index=pd.bdate_range("2019-01-01", periods=n))

    norm = DataProvider._normalise(df)
    # every column is plain numpy float64, and .values is a real ndarray
    assert all(str(norm[c].dtype) == "float64" for c in norm.columns)
    assert isinstance(norm["close"].values, np.ndarray)
    # the downstream that used to raise now runs cleanly
    detect_advanced(norm)
    signal_for_frame("X", norm)


def test_normalise_drops_na_rows_in_ohlc():
    df = pd.DataFrame({
        "open": [1.0, np.nan, 3.0], "high": [1.0, np.nan, 3.0],
        "low": [1.0, np.nan, 3.0], "close": [1.0, np.nan, 3.0],
        "volume": [10, 20, 30],
    }, index=pd.bdate_range("2024-01-01", periods=3))
    norm = DataProvider._normalise(df)
    assert len(norm) == 2          # the NaN row dropped
