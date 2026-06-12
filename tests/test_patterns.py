"""Tests for candlestick and chart pattern detectors."""

import pandas as pd

from nsetrade.indicators import add_all
from nsetrade.patterns.candlestick import detect_candlesticks
from nsetrade.patterns.chart import detect_chart_patterns, support_resistance


def _row(o, h, l, c, v=100000):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def test_bullish_engulfing_detected():
    # day1 red (10->9), day2 green engulfing (8.5 -> 10.5)
    rows = [
        _row(10, 10.2, 8.8, 9.0),      # red
        _row(8.5, 10.6, 8.4, 10.5),    # green engulfing prior body
    ]
    df = pd.DataFrame(rows, index=pd.bdate_range("2021-01-01", periods=2))
    flags = detect_candlesticks(df)
    assert bool(flags["bullish_engulfing"].iloc[-1])


def test_hammer_shape():
    # small body near top, long lower shadow, negligible upper shadow
    rows = [
        _row(10, 10.1, 9.9, 10.0),
        _row(10, 10.0, 8.5, 9.95),  # long lower shadow, tiny body, no upper wick
    ]
    df = pd.DataFrame(rows, index=pd.bdate_range("2021-01-01", periods=2))
    flags = detect_candlesticks(df)
    assert bool(flags["hammer"].iloc[-1])


def test_doji_detected():
    rows = [_row(10, 10.5, 9.5, 10.001)]
    df = pd.DataFrame(rows, index=pd.bdate_range("2021-01-01", periods=1))
    flags = detect_candlesticks(df)
    assert bool(flags["doji"].iloc[0])


def test_golden_cross_in_uptrend(make_ohlcv):
    import numpy as np
    # decline for 150 bars then a sustained rally: the 50-SMA will cross
    # above the 200-SMA once the rally matures -> a golden cross.
    prices = np.concatenate([
        np.linspace(100, 70, 250),   # long decline: both SMAs defined, 50<200
        np.linspace(70, 240, 200),   # strong rally: 50-SMA crosses above 200
    ])
    df = make_ohlcv(prices)
    enriched = add_all(df)
    flags = detect_chart_patterns(enriched)
    assert flags["golden_cross"].any()


def test_chart_flags_boolean(trending_df):
    enriched = add_all(trending_df)
    flags = detect_chart_patterns(enriched)
    assert flags.dtypes.eq(bool).all()


def test_support_resistance_levels(trending_df):
    sr = support_resistance(trending_df)
    assert "support" in sr and "resistance" in sr
    assert isinstance(sr["support"], list)
