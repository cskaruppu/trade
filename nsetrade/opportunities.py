"""Opportunity ranker — surface the most tradeable setups across a universe.

The screener ranks by raw signal score. This goes further: for each stock it
combines three independent, evidence-based ingredients into one **opportunity
score**, so the product can answer "which stocks are worth trading right now?"

  1. **Multi-timeframe conviction** — does daily/weekly/monthly agree? (trend)
  2. **Historical pattern edge** — has this stock's current chart pattern
     (cup & handle, etc.) actually *worked* in the past? (evidence)
  3. **Reward:risk** — does a sane trade plan offer a decent R:R? (quality)

Every component is kept on the result object, so a ranking is explainable — you
see *why* a stock scored, not just a number. This is technical probability, not
a profit guarantee; small pattern samples are noisy and markets change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# How strongly each evidence source pulls on the final score. Conviction is the
# base (already on the signal-score scale); edge and R:R are bonuses on top.
EDGE_WEIGHT = 3.0
RR_WEIGHT = 0.4
EDGE_MIN_SAMPLE = 5  # occurrences needed before an edge counts at full weight


@dataclass
class Opportunity:
    symbol: str
    score: float
    side: str
    close: float
    conviction: float
    aligned: str
    signal_verdict: str
    pattern: Optional[str] = None
    pattern_win_rate: Optional[float] = None
    pattern_occurrences: Optional[int] = None
    rr: Optional[float] = None
    reasons: str = ""

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 2),
            "close": round(self.close, 2),
            "conviction": round(self.conviction, 2),
            "aligned": self.aligned,
            "pattern": self.pattern or "-",
            "edge": (f"{self.pattern_win_rate:.0%}/{self.pattern_occurrences}"
                     if self.pattern_win_rate is not None else "-"),
            "rr": f"{self.rr:.1f}" if self.rr is not None else "-",
            "verdict": self.signal_verdict,
        }


@dataclass
class OpportunityResult:
    opportunities: list[Opportunity]
    errors: dict[str, str] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([o.as_row() for o in self.opportunities])


def score_opportunity(
    symbol: str,
    daily: pd.DataFrame,
    *,
    side: str = "long",
    with_edge: bool = True,
    capital: float = 100_000.0,
    risk_pct: float = 0.01,
) -> Opportunity:
    """Score a single stock from its daily OHLCV. Pure (no network) — testable.

    Raises if there aren't enough bars to compute a signal; callers in
    :func:`rank_opportunities` catch per-symbol errors.
    """
    from .confluence import confluence_for_frames
    from .patterns import detect_advanced
    from .resample import resample_ohlcv
    from .signals.engine import signal_for_frame

    dir_sign = 1.0 if side == "long" else -1.0
    sig = signal_for_frame(symbol, daily)

    # 1. multi-timeframe conviction (signed: + bullish, - bearish)
    frames = {tf: resample_ohlcv(daily, tf)
              for tf in ("daily", "weekly", "monthly")}
    conf = confluence_for_frames(symbol, frames)
    base = dir_sign * conf.conviction

    # 2. best pattern on this side + its historical edge
    pattern = win_rate = occurrences = None
    edge_bonus = 0.0
    want = "bullish" if side == "long" else "bearish"
    for m in detect_advanced(daily):
        if m.direction != want:
            continue
        e = None
        if with_edge:
            try:
                from .ai import _pattern_key
                from .edge import pattern_edge
                key = _pattern_key(m.name)
                if key:
                    e = pattern_edge(daily, key, forward_bars=20)
            except Exception:  # noqa: BLE001 - edge is best-effort
                e = None
        # prefer the pattern with the strongest evidence; fall back to any match
        cand_bonus = 0.0
        if e is not None and e.occurrences:
            cand_bonus = ((e.win_rate - 0.5) * 2.0
                          * min(e.occurrences / EDGE_MIN_SAMPLE, 1.0))
        if pattern is None or cand_bonus > edge_bonus:
            pattern = m.name
            edge_bonus = cand_bonus
            if e is not None and e.occurrences:
                win_rate, occurrences = e.win_rate, e.occurrences
            else:
                win_rate = occurrences = None

    # 3. reward:risk of a sane trade plan in this direction
    rr = None
    rr_bonus = 0.0
    try:
        from .tradeplan import trade_plan
        plan = trade_plan(symbol, daily, direction=side,
                          capital=capital, risk_pct=risk_pct)
        rr = plan.rr
        rr_bonus = max(0.0, min(rr - 1.0, 2.0))
    except Exception:  # noqa: BLE001
        pass

    score = base + EDGE_WEIGHT * edge_bonus + RR_WEIGHT * rr_bonus
    return Opportunity(
        symbol=symbol, score=score, side=side, close=sig.close,
        conviction=conf.conviction, aligned=conf.aligned,
        signal_verdict=sig.verdict, pattern=pattern,
        pattern_win_rate=win_rate, pattern_occurrences=occurrences, rr=rr,
        reasons="; ".join(sig.reasons),
    )


def rank_opportunities(
    symbols: list[str],
    *,
    provider: str = "yfinance",
    provider_config: Optional[dict] = None,
    period_days: int = 500,
    side: str = "long",
    with_edge: bool = True,
    top: int = 15,
    min_score: Optional[float] = None,
    on_progress=None,
    _provider_obj=None,
) -> OpportunityResult:
    """Fetch and score every symbol, then rank by opportunity score (desc).

    Per-symbol data errors are collected in ``errors`` rather than aborting.
    Pass ``_provider_obj`` to inject a provider in tests (skips the factory).
    """
    from .data import get_provider

    prov = _provider_obj or get_provider(provider, provider_config)
    out: list[Opportunity] = []
    errors: dict[str, str] = {}

    for i, sym in enumerate(symbols):
        try:
            daily = prov.history(sym, period_days=period_days)
            out.append(score_opportunity(sym, daily, side=side,
                                         with_edge=with_edge))
        except Exception as exc:  # noqa: BLE001 - keep scanning
            errors[sym] = str(exc)
        if on_progress:
            on_progress(i + 1, len(symbols), sym)

    out.sort(key=lambda o: o.score, reverse=True)
    if min_score is not None:
        out = [o for o in out if o.score >= min_score]
    if top:
        out = out[:top]
    return OpportunityResult(opportunities=out, errors=errors)
