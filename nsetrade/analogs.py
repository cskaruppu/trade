"""Historical chart analogs — "when this stock looked like it does now, what
happened next?"

Takes the current chart shape (the last ``window`` bars), searches the stock's
own history for the most *similar* past shapes, and reports what happened in the
``forward`` bars after each. The aggregate is an honest base rate, not a
prediction — and it shows the sample size so a 4-analog match isn't mistaken for
a 40-analog one.

Shape similarity is the Pearson correlation of the two normalized price windows
(scale- and level-invariant, so a ₹50 stock and a ₹5,000 move compare fairly).
Overlapping and near-duplicate windows are de-duplicated so the analogs are
genuinely distinct episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Analog:
    end_date: str
    similarity: float        # 0-1 (clamped correlation)
    forward_return: float    # return over `forward` bars after the window
    end_pos: int = -1        # integer bar index of the window's last bar


@dataclass
class AnalogResult:
    analogs: list[Analog] = field(default_factory=list)
    window: int = 0
    forward: int = 0
    n: int = 0
    avg_forward: float = 0.0
    median_forward: float = 0.0
    win_rate: float = 0.0
    note: str = ""

    def describe(self) -> str:
        if self.n == 0:
            return self.note or "no close analogs found"
        return (f"{self.n} analogs · next {self.forward} bars averaged "
                f"{self.avg_forward:+.1%}, positive {self.win_rate:.0%} of the time")


def _zscore(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / (s if s else 1.0)


def aligned_paths(df: pd.DataFrame, result: "AnalogResult") -> list[dict]:
    """Aligned, rebased price paths for overlaying analogs on one mini-chart.

    Every path is indexed to 100 at ``x = 0`` (the window's last bar), so the
    shapes overlay regardless of price level. ``x`` runs from ``-(window-1)`` to
    ``+forward``. The first entry (``kind == "current"``) is today's window and
    has no forward data; each analog adds its window *and* the forward bars that
    actually followed it. Use it to literally see "these past shapes matched, and
    here's where they went."
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    w, f = result.window, result.forward
    paths: list[dict] = []

    def _rebase(seg, x0):
        base = seg[w - 1]
        if base <= 0:
            return None
        return {"x": np.arange(x0, x0 + len(seg)),
                "y": seg / base * 100.0}

    cur = _rebase(close[-w:], -(w - 1))
    if cur:
        cur.update(kind="current", similarity=1.0, label="now")
        paths.append(cur)

    for a in result.analogs:
        i = a.end_pos
        if i < w - 1 or i + f >= n:
            continue
        seg = close[i - w + 1:i + f + 1]
        p = _rebase(seg, -(w - 1))
        if p:
            p.update(kind="analog", similarity=a.similarity, label=a.end_date)
            paths.append(p)
    return paths


def find_analogs(df: pd.DataFrame, *, window: int = 40, forward: int = 30,
                 top_k: int = 8, min_similarity: float = 0.7,
                 min_gap: Optional[int] = None) -> AnalogResult:
    """Find the most similar past ``window``-bar shapes to the current one.

    Returns up to ``top_k`` distinct analogs (each ≥ ``min_gap`` bars apart, and
    not overlapping the current window) with similarity ≥ ``min_similarity``,
    plus aggregate forward-return stats over ``forward`` bars.
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    min_gap = min_gap or window
    if n < 2 * window + forward:
        return AnalogResult(window=window, forward=forward,
                            note="not enough history for analogs")

    cur_z = _zscore(close[-window:])
    last_i = min(n - window - 1, n - forward - 1)   # no overlap + has forward data
    cands = []
    for i in range(window - 1, last_i + 1):
        z = _zscore(close[i - window + 1:i + 1])
        sim = float(np.corrcoef(cur_z, z)[0, 1])
        if np.isnan(sim) or sim < min_similarity:
            continue
        if close[i] <= 0:
            continue
        fwd = close[i + forward] / close[i] - 1.0
        cands.append((i, sim, fwd))

    cands.sort(key=lambda c: c[1], reverse=True)
    picked: list[tuple] = []
    for i, sim, fwd in cands:
        if all(abs(i - j) >= min_gap for j, _, _ in picked):
            picked.append((i, sim, fwd))
        if len(picked) >= top_k:
            break

    if not picked:
        return AnalogResult(window=window, forward=forward,
                            note=f"no past shape ≥ {min_similarity:.0%} similar")

    picked.sort(key=lambda c: c[1], reverse=True)
    analogs = [Analog(str(df.index[i].date()),
                      max(0.0, min(1.0, sim)), fwd, end_pos=i)
               for i, sim, fwd in picked]
    fwds = np.array([a.forward_return for a in analogs])
    return AnalogResult(
        analogs=analogs, window=window, forward=forward, n=len(analogs),
        avg_forward=float(fwds.mean()), median_forward=float(np.median(fwds)),
        win_rate=float((fwds > 0).mean()))
