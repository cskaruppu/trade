"""Explain *why* a pattern was drawn — the analysis basis, in plain terms.

Turns a :class:`PatternMatch` (plus its frame) into a structured, honest
explanation a user can read and confirm: what the pattern is, the specific
criteria that were checked (with the actual measured numbers and pass/fail), the
**downside / invalidation** (the level that negates it), and the measured-move
target. This is derived from the real detection — not AI narration — so it's
deterministic and testable. An AI layer can add prose on top, but the facts here
stand on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Criterion:
    label: str
    ok: Optional[bool]      # True pass / False fail / None informational
    detail: str


@dataclass
class BreakoutCheck:
    state: str            # confirmed | volume_light | approaching | far | none
    emoji: str
    message: str
    level: Optional[float] = None
    pct_to_level: Optional[float] = None   # last close vs the trigger (signed)
    volume_ok: Optional[bool] = None


def breakout_check(df: pd.DataFrame, match, *, near_pct: float = 0.03,
                   vol_lookback: int = 50, vol_mult: float = 1.3) -> BreakoutCheck:
    """Has price *actually* broken out — a decisive close above the trigger, on
    above-average volume — or is it still just approaching it?

    Guards against the common false start of buying a trendline poke while the
    real horizontal resistance (the rim/pivot) is still overhead. ``near_pct`` is
    how close (below) counts as "approaching".
    """
    level = getattr(match, "breakout_level", None)
    if df is None or not len(df) or not level or level <= 0:
        return BreakoutCheck("none", "•", "no breakout level to check")
    last = float(df["close"].iloc[-1])
    pct = last / level - 1.0
    from .patterns.advanced import volume_confirms
    vok = volume_confirms(df, lookback=vol_lookback, mult=vol_mult)
    above = last >= level

    if above and vok:
        return BreakoutCheck(
            "confirmed", "✅",
            f"Confirmed — closed ₹{last:.1f}, above the ₹{level:.1f} trigger on "
            "above-average volume.", level, pct, True)
    if above and vok is False:
        return BreakoutCheck(
            "volume_light", "⚠️",
            f"Above the ₹{level:.1f} trigger (₹{last:.1f}) but on light volume — "
            "breakouts without a volume push often fail back. Wait for confirmation.",
            level, pct, False)
    if above:      # vok is None (not enough volume history) — treat as tentative
        return BreakoutCheck(
            "confirmed", "✅",
            f"Closed above the ₹{level:.1f} trigger (₹{last:.1f}).", level, pct, None)
    if pct >= -near_pct:
        return BreakoutCheck(
            "approaching", "⏳",
            f"Approaching — ₹{last:.1f} is {abs(pct):.1%} below the ₹{level:.1f} "
            "trigger. Wait for a decisive close ABOVE it (ideally on volume); don't "
            "jump early on a trendline poke.", level, pct, vok)
    return BreakoutCheck(
        "far", "🔶",
        f"Still building — ₹{last:.1f} is {abs(pct):.1%} below the ₹{level:.1f} "
        "breakout trigger.", level, pct, vok)


@dataclass
class PatternExplanation:
    name: str
    basis: str                                   # one-line description of the method
    criteria: list[Criterion] = field(default_factory=list)
    stop: Optional[float] = None                 # invalidation level
    invalidation: str = ""
    target: Optional[float] = None               # measured-move target
    upside_pct: Optional[float] = None
    downside_pct: Optional[float] = None


# what each pattern *is* and the basis on which it's drawn
_BASIS = {
    "cup_and_handle": "A rounded 'U' base between two rims of similar height, then "
                      "a shallow handle pullback; the breakout is a push above the "
                      "rim. Basis: a completed accumulation cycle resuming its trend.",
    "vcp": "A series of progressively tighter pullbacks as the stock coils near its "
           "highs (supply drying up), then a break above the last pivot. Basis: "
           "volatility contraction preceding expansion.",
    "rounding_bottom": "A deep, long, U-shaped 'saucer' base that gradually recovers "
                       "and reclaims a horizontal resistance near the prior high. "
                       "Basis: a multi-quarter accumulation/turnaround completing.",
    "flat_base": "A shallow, tight sideways shelf that forms *after* an advance, then "
                 "a breakout above the shelf. Basis: the stock digesting gains before "
                 "continuing higher.",
    "darvas_box": "A tight high/low box the price consolidates in, then closes above "
                  "the box top. Basis: a range breakout with the box floor as risk.",
    "flag": "A sharp advance (the pole) then a small, orderly pullback (the flag), "
            "resolving upward. Basis: a brief pause inside a strong move.",
    "double_bottom": "Two similar lows with a peak between (a 'W'); the breakout is a "
                     "push above that middle peak (the neckline). Basis: a failed "
                     "retest of the low signalling a reversal.",
    "accumulation": "A flat support zone under a falling resistance line after a "
                    "decline; the breakout clears the trendline. Basis: quiet "
                    "accumulation ending a downtrend.",
}

_GENERIC_BASIS = ("A structural formation in the recent range; the breakout level is "
                  "the resistance to clear and the support is the risk level.")


def _rsi_last(df: pd.DataFrame) -> Optional[float]:
    try:
        from .indicators import add_all
        s = add_all(df)["rsi_14"].dropna()
        return float(s.iloc[-1]) if len(s) else None
    except Exception:  # noqa: BLE001
        return None


def explain_pattern(df: pd.DataFrame, match, key: Optional[str] = None) -> PatternExplanation:
    """Build a plain-language, evidence-based explanation of ``match``.

    ``key`` is the detector key (e.g. ``"cup_and_handle"``); if omitted the
    explanation still works with generic criteria.
    """
    name = getattr(match, "name", "Pattern")
    basis = _BASIS.get(key or "", _GENERIC_BASIS)
    ex = PatternExplanation(name=name, basis=basis)
    if df is None or not len(df):
        return ex

    last = float(df["close"].iloc[-1])
    support = getattr(match, "support", None)
    breakout = getattr(match, "breakout_level", None)
    status = getattr(match, "status", "")

    # base duration (bars from the pattern start to now)
    start = getattr(match, "start", None)
    if start is not None:
        try:
            dur = int(df.index.get_indexer([start], method="nearest")[0])
            dur = len(df) - 1 - dur
            ex.criteria.append(Criterion(
                "Base duration", dur >= 15,
                f"{dur} bars — {'a proper base' if dur >= 15 else 'short; less reliable'}"))
        except Exception:  # noqa: BLE001
            pass

    # base depth / tightness
    if support and breakout and breakout > 0:
        depth = (breakout - support) / breakout
        ex.criteria.append(Criterion(
            "Base depth", depth <= 0.40,
            f"{depth:.0%} from support to breakout"
            f" — {'reasonable' if depth <= 0.40 else 'deep/loose'}"))

    # breakout vs still forming
    if breakout:
        broke = status == "breakout"
        ex.criteria.append(Criterion(
            "Breakout", broke,
            (f"price {last:.1f} has cleared {breakout:.1f}" if broke
             else f"still forming — needs a close above {breakout:.1f}")))

    # volume confirmation
    vc = getattr(match, "volume_confirmed", None)
    if vc is not None:
        ex.criteria.append(Criterion(
            "Volume", bool(vc),
            "breakout backed by above-average volume" if vc
            else "volume did not surge on the breakout — weaker"))

    # momentum context
    rsi = _rsi_last(df)
    if rsi is not None:
        ex.criteria.append(Criterion(
            "Momentum (RSI)", None,
            f"RSI {rsi:.0f}" + (" — overbought, expect pullbacks" if rsi >= 70
                                else " — weak" if rsi <= 40 else " — constructive")))

    # the detector's own measurement note
    note = getattr(match, "note", "")
    if note:
        ex.criteria.append(Criterion("Pattern measurements", None, note))

    # downside / invalidation
    if support:
        ex.stop = float(support)
        ex.downside_pct = (support / last - 1.0) if last else None
        ex.invalidation = (
            f"A close back below ₹{support:.1f} negates this base — that's the line "
            f"in the sand ({ex.downside_pct:+.0%} from ₹{last:.1f}).")

    # measured-move target
    if support and breakout and breakout > support:
        target = breakout + (breakout - support)
        ex.target = float(target)
        ex.upside_pct = (target / last - 1.0) if last else None

    return ex
