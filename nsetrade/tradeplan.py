"""Auto-generate a concrete trade plan from a signal or pattern.

Given the current price, an ATR (for volatility-based stops) and optionally a
detected pattern (for a measured-move target), this produces an actionable plan:
entry, stop-loss, target, risk:reward and a position size derived from your
capital and the fraction you're willing to risk per trade.

This is decision *support*, not advice — you place and manage the orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .indicators import add_all, atr as atr_indicator
from .patterns.advanced import PatternMatch


@dataclass
class TradePlan:
    symbol: str
    direction: str            # "long" | "short"
    entry: float
    stop: float
    target: float
    shares: int
    risk_per_share: float
    risk_amount: float
    reward_amount: float
    rr: float                 # reward-to-risk ratio
    capital: float
    risk_pct: float
    rationale: str = ""

    def describe(self) -> str:
        return (
            f"Trade plan — {self.symbol} ({self.direction})\n"
            f"  Entry            : {self.entry:.2f}\n"
            f"  Stop-loss        : {self.stop:.2f}  "
            f"(risk {self.risk_per_share:.2f}/share)\n"
            f"  Target           : {self.target:.2f}\n"
            f"  Reward:Risk      : {self.rr:.2f} : 1\n"
            f"  Position size    : {self.shares} shares "
            f"(₹{self.shares * self.entry:,.0f})\n"
            f"  Risk on trade    : ₹{self.risk_amount:,.0f} "
            f"({self.risk_pct:.1%} of ₹{self.capital:,.0f})\n"
            + (f"  Rationale        : {self.rationale}\n" if self.rationale else "")
        )


def trade_plan(
    symbol: str,
    df: pd.DataFrame,
    *,
    direction: str = "long",
    capital: float = 100_000.0,
    risk_pct: float = 0.01,
    stop_atr: float = 2.0,
    rr_target: float = 2.0,
    pattern: Optional[PatternMatch] = None,
    atr_period: int = 14,
) -> TradePlan:
    """Build a trade plan for the latest bar of ``df``.

    The stop is ``stop_atr`` ATRs from entry. The target is either the pattern's
    measured move (if a ``pattern`` with support/resistance is supplied) or a
    fixed reward:risk multiple (``rr_target``). Position size risks ``risk_pct``
    of ``capital``.
    """
    if len(df) < atr_period + 2:
        raise ValueError("not enough bars for a trade plan")

    enriched = add_all(df)
    atr = float(atr_indicator(enriched, atr_period).iloc[-1])
    entry = float(df["close"].iloc[-1])
    if atr <= 0:
        atr = max(entry * 0.01, 0.05)

    long = direction == "long"
    stop = entry - stop_atr * atr if long else entry + stop_atr * atr
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        raise ValueError("degenerate stop distance")

    # target: measured move from the pattern if available, else RR multiple.
    # Only use the measured move if it sits *beyond* entry in the trade's
    # direction — once price has run past the breakout level it would otherwise
    # produce a target behind us.
    rationale = f"{stop_atr}×ATR stop"
    target = None
    if pattern is not None and pattern.breakout_level and pattern.support:
        height = abs(pattern.breakout_level - pattern.support)
        candidate = (pattern.breakout_level + height if long
                     else pattern.breakout_level - height)
        beyond_entry = candidate > entry if long else candidate < entry
        if beyond_entry:
            target = candidate
            rationale = f"{pattern.name} measured move ({height:.1f} pts)"
    if target is None:
        target = entry + rr_target * risk_per_share if long else \
            entry - rr_target * risk_per_share
        rationale += f", {rr_target}:1 target"

    reward_per_share = abs(target - entry)
    rr = reward_per_share / risk_per_share if risk_per_share else 0.0

    risk_amount = capital * risk_pct
    shares = int(risk_amount / risk_per_share)
    # never deploy more than the whole account
    shares = min(shares, int(capital / entry)) if entry > 0 else 0

    return TradePlan(
        symbol=symbol,
        direction=direction,
        entry=round(entry, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        shares=shares,
        risk_per_share=round(risk_per_share, 2),
        risk_amount=round(shares * risk_per_share, 2),
        reward_amount=round(shares * reward_per_share, 2),
        rr=round(rr, 2),
        capital=capital,
        risk_pct=risk_pct,
        rationale=rationale,
    )
