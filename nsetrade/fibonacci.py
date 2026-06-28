"""Fibonacci retracement and trend-based extension — auto-computed.

TradingView gives you a *drawing tool*; you pick the swing by hand. In an
analytical product we instead detect the dominant swing automatically and
report the standard Fibonacci levels, where price currently sits, and the
nearest level acting as support/resistance. This is the industry-standard
ratio set used by technical traders.

  * **Retracement** (0 / 23.6 / 38.2 / 50 / 61.8 / 78.6 / 100 %) — pullback
    levels within the prior swing; common entry/support zones.
  * **Trend-based extension** (1.0 / 1.272 / 1.618 / 2.0 / 2.618) — projected
    targets beyond the swing, used for profit-taking.

Levels are geometry on a chosen swing, not predictions; price respects them
often enough to be useful, not always.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

RETRACEMENT_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
EXTENSION_RATIOS = [0.618, 1.0, 1.272, 1.618, 2.0, 2.618]


@dataclass
class FibLevel:
    ratio: float
    price: float

    @property
    def label(self) -> str:
        return f"{self.ratio * 100:.1f}%"


@dataclass
class FibRetracement:
    found: bool
    direction: str = ""          # "up" (levels are support) | "down" (resistance)
    swing_high: float = 0.0
    swing_low: float = 0.0
    levels: list[FibLevel] = field(default_factory=list)
    current_price: float = 0.0
    nearest: FibLevel | None = None
    note: str = ""

    def describe(self) -> str:
        if not self.found:
            return "Fib retracement: no clear swing"
        role = "support" if self.direction == "up" else "resistance"
        near = (f" — nearest {self.nearest.label} @ {self.nearest.price:.1f} "
                f"({role})") if self.nearest else ""
        return (f"Fib retracement [{self.direction}] "
                f"{self.swing_low:.1f}–{self.swing_high:.1f}{near}")


@dataclass
class FibExtension:
    found: bool
    levels: list[FibLevel] = field(default_factory=list)
    point_a: float = 0.0          # swing start
    point_b: float = 0.0          # swing end
    point_c: float = 0.0          # retracement pivot (projection origin)
    note: str = ""

    def describe(self) -> str:
        if not self.found:
            return "Fib extension: no A-B-C structure"
        tgts = ", ".join(f"{l.label}@{l.price:.1f}" for l in self.levels
                         if l.ratio >= 1.0)
        return f"Fib extension targets: {tgts}"


def fib_retracement(df: pd.DataFrame, *, lookback: int = 120) -> FibRetracement:
    """Auto Fibonacci retracement over the dominant swing in ``lookback`` bars.

    The dominant swing is the window's extreme high and low; their order in
    time sets the direction. Up-swing (low then high) → levels measured down
    from the high act as support; down-swing → measured up from the low act as
    resistance.
    """
    win = df.tail(lookback)
    if len(win) < 20:
        return FibRetracement(found=False, note="not enough bars")
    hi = float(win["high"].max())
    lo = float(win["low"].min())
    if hi <= lo:
        return FibRetracement(found=False)
    hi_at = win["high"].idxmax()
    lo_at = win["low"].idxmin()
    up = hi_at > lo_at                      # high came after low → uptrend swing
    span = hi - lo
    levels = []
    for r in RETRACEMENT_RATIOS:
        price = hi - span * r if up else lo + span * r
        levels.append(FibLevel(r, price))
    current = float(win["close"].iloc[-1])
    nearest = min(levels, key=lambda L: abs(L.price - current))
    return FibRetracement(
        found=True, direction="up" if up else "down",
        swing_high=hi, swing_low=lo, levels=levels,
        current_price=current, nearest=nearest,
        note=f"swing {lo:.1f}-{hi:.1f}")


def fib_extension(df: pd.DataFrame, *, lookback: int = 160) -> FibExtension:
    """Trend-based Fibonacci extension from an auto-detected A-B-C structure.

    A = a swing low, B = the swing high after it, C = the pullback low after B.
    Targets are projected as ``C + ratio * (B - A)`` — the classic upside
    profit objectives. Returns ``found=False`` if no clean A-B-C is present.
    """
    from .patterns.advanced import _swings

    win = df.tail(lookback)
    if len(win) < 30:
        return FibExtension(found=False, note="not enough bars")
    highs, lows = _swings(win)
    if not highs or len(lows) < 2:
        return FibExtension(found=False, note="no swings")

    hv = win["high"].values
    lv = win["low"].values
    # B = most recent swing high
    b_idx = highs[-1]
    # A = lowest swing low before B
    prior_lows = [i for i in lows if i < b_idx]
    if not prior_lows:
        return FibExtension(found=False, note="no swing low before high")
    a_idx = min(prior_lows, key=lambda i: lv[i])
    # C = lowest swing low after B (the retracement); fall back to last low
    later_lows = [i for i in lows if i > b_idx]
    c_idx = min(later_lows, key=lambda i: lv[i]) if later_lows else None

    a, b = float(lv[a_idx]), float(hv[b_idx])
    if b <= a:
        return FibExtension(found=False, note="degenerate A-B")
    c = float(lv[c_idx]) if c_idx is not None else float(win["close"].iloc[-1])
    span = b - a
    levels = [FibLevel(r, c + span * r) for r in EXTENSION_RATIOS]
    return FibExtension(found=True, levels=levels, point_a=a, point_b=b, point_c=c,
                        note=f"A {a:.1f} → B {b:.1f} → C {c:.1f}")
