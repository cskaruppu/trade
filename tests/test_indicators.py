"""Tests for technical indicators against known properties / values."""

import numpy as np
import pandas as pd

from nsetrade.indicators import core


def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = core.sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 50, dtype=float))
    r = core.rsi(s, 14)
    assert r.dropna().iloc[-1] == 100.0


def test_rsi_bounds(trending_df):
    r = core.rsi(trending_df["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_columns(trending_df):
    m = core.macd(trending_df["close"])
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]
    # histogram == macd - signal
    diff = (m["macd"] - m["macd_signal"]) - m["macd_hist"]
    assert diff.abs().dropna().max() < 1e-9


def test_atr_positive(trending_df):
    a = core.atr(trending_df, 14).dropna()
    assert (a > 0).all()


def test_bollinger_contains_price_mostly(trending_df):
    bb = core.bollinger_bands(trending_df["close"], 20, 2)
    close = trending_df["close"]
    inside = ((close <= bb["bb_upper"]) & (close >= bb["bb_lower"])).dropna()
    # Most points sit inside the bands; a trending series rides them so the
    # containment ratio is below the ~95% seen on stationary data.
    assert inside.mean() > 0.5


def test_adx_range(trending_df):
    a = core.adx(trending_df, 14)["adx"].dropna()
    assert (a >= 0).all() and (a <= 100).all()


def test_add_all_columns(trending_df):
    out = core.add_all(trending_df)
    for col in ["sma_50", "sma_200", "rsi_14", "macd", "bb_upper",
                "adx", "atr_14", "obv"]:
        assert col in out.columns
