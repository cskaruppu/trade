"""Standalone support / resistance trendlines for any chart.

Independent of the pattern detectors: fit a straight line through the recent
swing highs (resistance) and swing lows (support) and project it to the latest
bar. Useful on its own to see the prevailing up/down channel — the line a
chartist draws across the highs or lows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fit(idx, positions, prices, n_swings):
    pts = positions[-n_swings:]
    if len(pts) < 2:
        return None
    x = np.array(pts, float)
    y = np.array([prices[i] for i in pts], float)
    slope, intercept = np.polyfit(x, y, 1)
    last = len(idx) - 1
    x0 = pts[0]
    return {
        "x": [idx[x0], idx[last]],
        "y": [float(slope * x0 + intercept), float(slope * last + intercept)],
        "slope": float(slope),
        "direction": ("down" if slope < 0 else "up" if slope > 0 else "flat"),
    }


def fit_trendlines(df: pd.DataFrame, *, n_swings: int = 3,
                   left: int = 3, right: int = 3) -> dict:
    """Return ``{"resistance": {...}, "support": {...}}`` line endpoints.

    Each line: ``x`` (two dates), ``y`` (two prices), ``slope`` and a
    ``direction`` ("up"/"down"/"flat"). Either key may be absent if there aren't
    enough swing pivots.
    """
    from .patterns.advanced import _swings

    if len(df) < 30:
        return {}
    highs, lows = _swings(df, left=left, right=right)
    out: dict = {}
    res = _fit(df.index, highs, df["high"].values, n_swings)
    sup = _fit(df.index, lows, df["low"].values, n_swings)
    if res:
        out["resistance"] = res
    if sup:
        out["support"] = sup
    return out
