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


def _pattern_signals(
    df: pd.DataFrame,
    detector,
    *,
    forward_bars: int,
    min_history: int,
    step: int,
):
    """Walk history and yield one record per de-duplicated breakout.

    Each record is ``(i, fwd, mfe, mae)`` where ``i`` is the positional entry
    bar. The detector only ever sees data up to bar ``i`` (no look-ahead).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    out = []
    last_signal = -10_000
    end = n - forward_bars
    i = min_history
    while i < end:
        if i - last_signal < forward_bars:
            i += step
            continue
        try:
            m = detector(df.iloc[: i + 1])
        except Exception:  # noqa: BLE001
            i += step
            continue
        if m.found and m.status == "breakout":
            entry = close[i]
            if entry > 0:
                wc = close[i + 1: i + 1 + forward_bars]
                wh = high[i + 1: i + 1 + forward_bars]
                wl = low[i + 1: i + 1 + forward_bars]
                out.append((i, float(wc[-1] / entry - 1.0),
                            float(wh.max() / entry - 1.0),
                            float(wl.min() / entry - 1.0)))
                last_signal = i
        i += step
    return out


def _edge_from_records(pattern, records, forward_bars, target_pct) -> PatternEdge:
    if not records:
        return PatternEdge(pattern, 0, forward_bars, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, target_pct)
    rets = [r[1] for r in records]
    mfe = [r[2] for r in records]
    mae = [r[3] for r in records]
    arr = np.array(rets)
    return PatternEdge(
        pattern=pattern,
        occurrences=len(rets),
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
    records = _pattern_signals(df, ADVANCED_DETECTORS[pattern],
                               forward_bars=forward_bars,
                               min_history=min_history, step=step)
    return _edge_from_records(pattern, records, forward_bars, target_pct)


@dataclass
class ValidatedEdge:
    """In-sample vs out-of-sample edge — guards against curve-fitting.

    A pattern that only "works" on the data it was tuned on is a mirage. We
    split history chronologically: the older portion is in-sample, the recent
    portion out-of-sample. An edge we trust holds up on data it never saw.
    """
    pattern: str
    full: PatternEdge
    in_sample: PatternEdge
    out_sample: PatternEdge
    robust: bool
    verdict: str

    def describe(self) -> str:
        return (
            f"{self.pattern}: {self.verdict}\n"
            f"  full  : {self.full.describe()}\n"
            f"  in-sample : {self.in_sample.win_rate:.0%} win, "
            f"{self.in_sample.avg_return:+.1%} avg ({self.in_sample.occurrences}x)\n"
            f"  out-sample: {self.out_sample.win_rate:.0%} win, "
            f"{self.out_sample.avg_return:+.1%} avg ({self.out_sample.occurrences}x)"
        )


def pattern_edge_validated(
    df: pd.DataFrame,
    pattern: str,
    *,
    forward_bars: int = 20,
    target_pct: float = 0.05,
    min_history: int = 80,
    step: int = 1,
    split: float = 0.7,
    min_oos: int = 3,
    degrade_tol: float = 0.5,
) -> ValidatedEdge:
    """Compute the edge, then re-check it out-of-sample.

    Signals are generated in a single look-ahead-free pass, then partitioned by
    whether the entry bar falls in the older ``split`` fraction (in-sample) or
    the recent remainder (out-of-sample). The edge is judged **robust** when the
    out-of-sample win-rate holds up (doesn't collapse by more than
    ``degrade_tol``) on at least ``min_oos`` independent out-of-sample signals.
    """
    if pattern not in ADVANCED_DETECTORS:
        raise ValueError(f"unknown pattern {pattern!r}")
    records = _pattern_signals(df, ADVANCED_DETECTORS[pattern],
                               forward_bars=forward_bars,
                               min_history=min_history, step=step)
    cut = int(len(df) * split)
    is_rec = [r for r in records if r[0] < cut]
    oos_rec = [r for r in records if r[0] >= cut]

    full = _edge_from_records(pattern, records, forward_bars, target_pct)
    in_s = _edge_from_records(pattern, is_rec, forward_bars, target_pct)
    oos = _edge_from_records(pattern, oos_rec, forward_bars, target_pct)

    if oos.occurrences < min_oos:
        robust, verdict = False, "unproven (too few out-of-sample signals)"
    elif in_s.occurrences == 0:
        robust, verdict = False, "no in-sample signals to compare"
    elif oos.win_rate >= in_s.win_rate * degrade_tol and oos.avg_return > 0:
        robust, verdict = True, "robust (edge held out-of-sample)"
    else:
        robust, verdict = False, "fragile (edge faded out-of-sample)"
    return ValidatedEdge(pattern, full, in_s, oos, robust, verdict)


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
