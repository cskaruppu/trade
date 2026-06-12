"""Candlestick reversal/continuation pattern detectors.

Each detector returns a boolean Series aligned to the input frame, ``True`` on
the bar where the pattern *completes*. Patterns are most meaningful at support
or resistance — the signal engine weighs them accordingly.
"""

from __future__ import annotations

import pandas as pd

# (key, human label, bias) — bias is +1 bullish, -1 bearish, 0 neutral
CANDLESTICK_PATTERNS = [
    ("doji", "Doji", 0),
    ("hammer", "Hammer", 1),
    ("inverted_hammer", "Inverted Hammer", 1),
    ("shooting_star", "Shooting Star", -1),
    ("hanging_man", "Hanging Man", -1),
    ("bullish_engulfing", "Bullish Engulfing", 1),
    ("bearish_engulfing", "Bearish Engulfing", -1),
    ("piercing_line", "Piercing Line", 1),
    ("dark_cloud_cover", "Dark Cloud Cover", -1),
    ("morning_star", "Morning Star", 1),
    ("evening_star", "Evening Star", -1),
]


def _bodies(df: pd.DataFrame):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0.0, pd.NA)
    upper_shadow = h - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - l
    return o, h, l, c, body, rng, upper_shadow, lower_shadow


def detect_candlesticks(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean DataFrame, one column per candlestick pattern."""
    o, h, l, c, body, rng, upper, lower = _bodies(df)
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_body = (prev_c - prev_o).abs()

    # small/large body relative to recent average range
    avg_range = (h - l).rolling(14, min_periods=3).mean()
    bullish = c > o
    bearish = c < o

    out = pd.DataFrame(index=df.index)

    # Doji: tiny body relative to range
    out["doji"] = (body <= 0.1 * rng).fillna(False)

    # Hammer: small body at top, long lower shadow, little upper shadow.
    # Use range-relative upper shadow so the rule holds for tiny bodies too.
    out["hammer"] = (
        (lower >= 2 * body) & (upper <= 0.15 * rng) & (body > 0)
    ).fillna(False)

    # Inverted hammer: long upper shadow, small lower shadow
    out["inverted_hammer"] = (
        (upper >= 2 * body) & (lower <= 0.15 * rng) & (body > 0)
    ).fillna(False)

    # Shooting star == inverted-hammer shape after an up move (bearish)
    up_context = c.shift(1) > c.shift(3)
    out["shooting_star"] = (out["inverted_hammer"] & up_context).fillna(False)

    # Hanging man == hammer shape after an up move (bearish)
    out["hanging_man"] = (out["hammer"] & up_context).fillna(False)

    # Bullish engulfing: today's green body engulfs yesterday's red body
    out["bullish_engulfing"] = (
        bearish.shift(1).fillna(False)
        & bullish
        & (c >= prev_o)
        & (o <= prev_c)
        & (body > prev_body)
    ).fillna(False)

    # Bearish engulfing: today's red body engulfs yesterday's green body
    out["bearish_engulfing"] = (
        bullish.shift(1).fillna(False)
        & bearish
        & (o >= prev_c)
        & (c <= prev_o)
        & (body > prev_body)
    ).fillna(False)

    # Piercing line: red then green closing above midpoint of prior body
    prev_mid = (prev_o + prev_c) / 2
    out["piercing_line"] = (
        bearish.shift(1).fillna(False)
        & bullish
        & (o < prev_c)
        & (c > prev_mid)
        & (c < prev_o)
    ).fillna(False)

    # Dark cloud cover: green then red closing below midpoint of prior body
    out["dark_cloud_cover"] = (
        bullish.shift(1).fillna(False)
        & bearish
        & (o > prev_c)
        & (c < prev_mid)
        & (c > prev_o)
    ).fillna(False)

    # Morning star: big red, small body, big green (3-bar bullish reversal)
    out["morning_star"] = (
        bearish.shift(2).fillna(False)
        & ((prev_c - prev_o).abs() <= 0.5 * avg_range.shift(1)).fillna(False)
        & bullish
        & (c > (o.shift(2) + c.shift(2)) / 2)
    ).fillna(False)

    # Evening star: big green, small body, big red (3-bar bearish reversal)
    out["evening_star"] = (
        bullish.shift(2).fillna(False)
        & ((prev_c - prev_o).abs() <= 0.5 * avg_range.shift(1)).fillna(False)
        & bearish
        & (c < (o.shift(2) + c.shift(2)) / 2)
    ).fillna(False)

    return out.fillna(False).astype(bool)
