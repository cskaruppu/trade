"""Regression: nullable (Float64/Int64) frames must not raise NA-ambiguous.

Reproduces the 'boolean value of NA is ambiguous' failure that hit the daily
timeframe when a provider returned pandas nullable dtypes, and asserts the
signal engine and detectors now coerce and survive it.
"""

import numpy as np
import pandas as pd

from nsetrade.patterns.advanced import detect_advanced
from nsetrade.signals.engine import signal_for_frame


def nullable_frame(n=320):
    idx = pd.bdate_range("2020-01-01", periods=n)
    base = 100 + np.cumsum(np.random.RandomState(0).normal(0, 1, n))
    df = pd.DataFrame({"open": base, "high": base + 1, "low": base - 1,
                       "close": base, "volume": [1e5] * n}, index=idx)
    # cast to pandas *nullable* dtypes and inject a genuine pd.NA row
    df = df.astype({"open": "Float64", "high": "Float64", "low": "Float64",
                    "close": "Float64", "volume": "Int64"})
    df.loc[df.index[5], "volume"] = pd.NA
    df.loc[df.index[7], "high"] = pd.NA
    return df


def test_signal_engine_handles_nullable_dtypes():
    sig = signal_for_frame("TEST", nullable_frame())   # must not raise
    assert sig.verdict


def test_detect_advanced_handles_nullable_dtypes():
    matches = detect_advanced(nullable_frame())         # must not raise
    assert isinstance(matches, list)
