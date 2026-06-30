"""Holding-period outlook — horizon-conditioned base rates from a stock's own past.

For the current chart shape (the last ``window`` bars), find the most similar
past episodes and report what happened over several *holding periods* (e.g.
3 months / 6 months / 1 year): the distribution of forward returns (median plus
a 25th–75th-percentile band), how often it was positive, the sample size, and
the implied **target-price band** from today's close.

This is the honest version of the analyst's "target X in 6 months": a base rate
with its sample size and a range of outcomes — not a point forecast. A wide band
or a small ``n`` is the signal to distrust it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .analogs import _zscore


@dataclass
class HorizonStat:
    label: str          # "3M", "6M", "1Y"
    bars: int           # forward bars this horizon represents
    n: int              # how many analogs had this much forward data
    median: float       # median forward return
    p25: float
    p75: float
    win_rate: float
    avg: float
    target_low: float   # today's price compounded by p25 / median / p75
    target_med: float
    target_high: float


@dataclass
class Outlook:
    price: float
    window: int
    horizons: list[HorizonStat] = field(default_factory=list)
    n_analogs: int = 0
    note: str = ""


def holding_period_outlook(df: pd.DataFrame, *, window: int = 40,
                           horizons: Optional[dict] = None,
                           min_similarity: float = 0.7, top_k: int = 40,
                           min_gap: Optional[int] = None) -> Outlook:
    """Base-rate outlook over several holding periods for the current shape.

    ``horizons`` maps a label to a number of forward bars, e.g.
    ``{"3M": 63, "6M": 126, "1Y": 252}`` for a daily frame. Uses up to ``top_k``
    distinct (non-overlapping) past windows similar to today's; each horizon's
    sample is the subset of those windows that have that much forward data, so a
    longer horizon honestly shows a smaller ``n``.
    """
    horizons = horizons or {"3M": 63, "6M": 126, "1Y": 252}
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    price = float(close[-1]) if n else 0.0
    min_gap = min_gap or window
    min_h = min(horizons.values())
    if n < 2 * window + min_h:
        return Outlook(price=price, window=window,
                       note="not enough history for an outlook")

    cur_z = _zscore(close[-window:])
    # candidates: a full window behind, not overlapping today, with ≥ shortest horizon ahead
    last_i = min(n - window - 1, n - 1 - min_h)
    cands = []
    for i in range(window - 1, last_i + 1):
        if close[i] <= 0:
            continue
        z = _zscore(close[i - window + 1:i + 1])
        sim = float(np.corrcoef(cur_z, z)[0, 1])
        if np.isnan(sim) or sim < min_similarity:
            continue
        cands.append((i, sim))

    cands.sort(key=lambda c: c[1], reverse=True)
    picked: list[tuple] = []
    for i, sim in cands:
        if all(abs(i - j) >= min_gap for j, _ in picked):
            picked.append((i, sim))
        if len(picked) >= top_k:
            break

    if not picked:
        return Outlook(price=price, window=window,
                       note=f"no past shape ≥ {min_similarity:.0%} similar")

    stats: list[HorizonStat] = []
    for label, h in sorted(horizons.items(), key=lambda kv: kv[1]):
        rets = [close[i + h] / close[i] - 1.0
                for i, _ in picked if i + h < n and close[i] > 0]
        if not rets:
            continue
        r = np.array(rets, dtype=float)
        med = float(np.median(r))
        p25, p75 = float(np.percentile(r, 25)), float(np.percentile(r, 75))
        stats.append(HorizonStat(
            label=label, bars=h, n=len(r), median=med, p25=p25, p75=p75,
            win_rate=float((r > 0).mean()), avg=float(r.mean()),
            target_low=price * (1 + p25), target_med=price * (1 + med),
            target_high=price * (1 + p75)))

    return Outlook(price=price, window=window, horizons=stats,
                   n_analogs=len(picked))
