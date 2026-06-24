"""Pattern *edge* backtesting — does a chart pattern actually work?

Most scanners show you a pattern but never tell you whether it has paid off in
the past. This module walks a stock's history, finds every prior occurrence of a
pattern's breakout, and measures what happened next: hit-rate, average forward
return, and how often it reached a useful move. That turns a detection from
"here's a shape" into "here's the historical edge of this shape on this stock".

The numbers are descriptive statistics on past data — **not** a promise about
the future, and small sample sizes (few occurrences) are noisy. Always read the
occurrence count alongside the hit-rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .patterns.advanced import ADVANCED_DETECTORS


@dataclass
class PatternEdge:
    pattern: str
    occurrences: int
    forward_bars: int
    win_rate: float                 # fraction with positive forward return
    avg_return: float
    median_return: float
    avg_max_favorable: float        # avg best gain reached within the window
    avg_max_adverse: float          # avg worst drawdown within the window
    hit_target_rate: float          # fraction that gained >= target_pct
    target_pct: float
    samples: list[float] = field(default_factory=list, repr=False)

    def describe(self) -> str:
        if self.occurrences == 0:
            return f"{self.pattern}: no past breakouts found"
        conf = "low sample" if self.occurrences < 5 else ""
        return (
            f"{self.pattern}: {self.occurrences} past breakouts → "
            f"{self.win_rate:.0%} positive after {self.forward_bars} bars, "
            f"avg {self.avg_return:+.1%} "
            f"(>{self.target_pct:.0%} move {self.hit_target_rate:.0%} of the time)"
            + (f"  [{conf}]" if conf else "")
        )


def pattern_edge(
    df: pd.DataFrame,
    pattern: str,
    *,
    forward_bars: int = 20,
    target_pct: float = 0.05,
    min_history: int = 80,
    step: int = 1,
) -> PatternEdge:
    """Measure the historical forward performance of ``pattern`` breakouts.

    For each historical bar we re-run the detector on data *up to that bar* and,
    when it reports a fresh breakout, record the forward return over the next
    ``forward_bars`` bars. Overlapping signals are de-duplicated with a cooldown
    equal to ``forward_bars`` so each move is counted once.
    """
    if pattern not in ADVANCED_DETECTORS:
        raise ValueError(
            f"unknown pattern {pattern!r}. Available: "
            f"{', '.join(ADVANCED_DETECTORS)}"
        )
    detector = ADVANCED_DETECTORS[pattern]
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    rets: list[float] = []
    mfe: list[float] = []
    mae: list[float] = []
    last_signal = -10_000

    end = n - forward_bars
    i = min_history
    while i < end:
        if i - last_signal < forward_bars:
            i += step
            continue
        sub = df.iloc[: i + 1]
        try:
            m = detector(sub)
        except Exception:  # noqa: BLE001
            i += step
            continue
        if m.found and m.status == "breakout":
            entry = close[i]
            if entry > 0:
                window_close = close[i + 1: i + 1 + forward_bars]
                window_high = high[i + 1: i + 1 + forward_bars]
                window_low = low[i + 1: i + 1 + forward_bars]
                fwd = window_close[-1] / entry - 1.0
                rets.append(float(fwd))
                mfe.append(float(window_high.max() / entry - 1.0))
                mae.append(float(window_low.min() / entry - 1.0))
                last_signal = i
        i += step

    occ = len(rets)
    if occ == 0:
        return PatternEdge(pattern, 0, forward_bars, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, target_pct)
    arr = np.array(rets)
    return PatternEdge(
        pattern=pattern,
        occurrences=occ,
        forward_bars=forward_bars,
        win_rate=float((arr > 0).mean()),
        avg_return=float(arr.mean()),
        median_return=float(np.median(arr)),
        avg_max_favorable=float(np.mean(mfe)),
        avg_max_adverse=float(np.mean(mae)),
        hit_target_rate=float((arr >= target_pct).mean()),
        target_pct=target_pct,
        samples=rets,
    )


def all_pattern_edges(
    df: pd.DataFrame,
    *,
    forward_bars: int = 20,
    target_pct: float = 0.05,
) -> list[PatternEdge]:
    """Run :func:`pattern_edge` for every bullish structural pattern."""
    out = []
    for key in ADVANCED_DETECTORS:
        try:
            out.append(pattern_edge(df, key, forward_bars=forward_bars,
                                    target_pct=target_pct))
        except Exception:  # noqa: BLE001
            continue
    return out
