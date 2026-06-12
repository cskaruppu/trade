"""Chart-level patterns and crossovers built on top of indicators.

These return a boolean DataFrame aligned to the input frame. They expect the
frame to already carry the indicator columns produced by
:func:`nsetrade.indicators.add_all` (the signal engine guarantees this).
"""

from __future__ import annotations

import pandas as pd

# (key, human label, bias)
CHART_PATTERNS = [
    ("golden_cross", "Golden Cross (50/200)", 1),
    ("death_cross", "Death Cross (50/200)", -1),
    ("macd_bull_cross", "MACD Bullish Crossover", 1),
    ("macd_bear_cross", "MACD Bearish Crossover", -1),
    ("breakout_20d", "20-Day Breakout (vol-confirmed)", 1),
    ("breakdown_20d", "20-Day Breakdown", -1),
    ("rsi_oversold_turn", "RSI Oversold Reversal", 1),
    ("rsi_overbought_turn", "RSI Overbought Reversal", -1),
    ("near_52w_high", "Near 52-Week High", 1),
    ("near_52w_low", "Near 52-Week Low", -1),
    ("bb_lower_tag", "Bollinger Lower-Band Tag", 1),
    ("bb_upper_tag", "Bollinger Upper-Band Tag", -1),
]


def _cross_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def _cross_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def support_resistance(
    df: pd.DataFrame, lookback: int = 60, num_levels: int = 3
) -> dict:
    """Return recent support/resistance levels from swing pivots.

    A swing high/low is a bar whose high/low is the extreme of a small window.
    Levels are the most recent distinct pivots within ``lookback`` bars.
    """
    window = df.tail(lookback)
    highs, lows = [], []
    h, l = window["high"].values, window["low"].values
    for i in range(2, len(window) - 2):
        if h[i] == max(h[i - 2 : i + 3]):
            highs.append(float(h[i]))
        if l[i] == min(l[i - 2 : i + 3]):
            lows.append(float(l[i]))
    resistance = sorted(set(round(x, 2) for x in highs), reverse=True)[:num_levels]
    support = sorted(set(round(x, 2) for x in lows))[-num_levels:]
    return {"support": support, "resistance": resistance}


def detect_chart_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect crossovers, breakouts and band/level interactions.

    ``df`` must already contain indicator columns (sma_50, sma_200, macd,
    macd_signal, rsi_14, bb_upper, bb_lower, vol_sma_20).
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    # SMA crosses
    if {"sma_50", "sma_200"}.issubset(df.columns):
        out["golden_cross"] = _cross_up(df["sma_50"], df["sma_200"])
        out["death_cross"] = _cross_down(df["sma_50"], df["sma_200"])
    else:  # pragma: no cover - defensive
        out["golden_cross"] = False
        out["death_cross"] = False

    # MACD crosses
    if {"macd", "macd_signal"}.issubset(df.columns):
        out["macd_bull_cross"] = _cross_up(df["macd"], df["macd_signal"])
        out["macd_bear_cross"] = _cross_down(df["macd"], df["macd_signal"])
    else:  # pragma: no cover
        out["macd_bull_cross"] = False
        out["macd_bear_cross"] = False

    # N-day breakout (Donchian-style), volume-confirmed
    hh20 = df["high"].rolling(20, min_periods=20).max().shift(1)
    ll20 = df["low"].rolling(20, min_periods=20).min().shift(1)
    vol_ok = True
    if "vol_sma_20" in df.columns:
        vol_ok = df["volume"] > 1.2 * df["vol_sma_20"]
    out["breakout_20d"] = ((close > hh20) & vol_ok).fillna(False)
    out["breakdown_20d"] = (close < ll20).fillna(False)

    # RSI reversals: crossing back above 30 / below 70
    if "rsi_14" in df.columns:
        rsi_ = df["rsi_14"]
        out["rsi_oversold_turn"] = (
            (rsi_ > 30) & (rsi_.shift(1) <= 30)
        ).fillna(False)
        out["rsi_overbought_turn"] = (
            (rsi_ < 70) & (rsi_.shift(1) >= 70)
        ).fillna(False)
    else:  # pragma: no cover
        out["rsi_oversold_turn"] = False
        out["rsi_overbought_turn"] = False

    # 52-week (252-bar) high/low proximity
    hh_52 = close.rolling(252, min_periods=50).max()
    ll_52 = close.rolling(252, min_periods=50).min()
    out["near_52w_high"] = (close >= 0.98 * hh_52).fillna(False)
    out["near_52w_low"] = (close <= 1.02 * ll_52).fillna(False)

    # Bollinger band tags
    if {"bb_upper", "bb_lower"}.issubset(df.columns):
        out["bb_lower_tag"] = (df["low"] <= df["bb_lower"]).fillna(False)
        out["bb_upper_tag"] = (df["high"] >= df["bb_upper"]).fillna(False)
    else:  # pragma: no cover
        out["bb_lower_tag"] = False
        out["bb_upper_tag"] = False

    return out.fillna(False).astype(bool)
