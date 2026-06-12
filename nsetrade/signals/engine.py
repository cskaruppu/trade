"""The signal engine.

Takes an OHLCV frame, layers on indicators and pattern detectors, then scores
the *latest* bar into a single bullish/bearish signal with human-readable
reasons. No single pattern is trusted alone — the score is a weighted sum, so
confirmation across signals is what produces a strong reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..data import get_provider
from ..indicators import add_all
from ..patterns.candlestick import CANDLESTICK_PATTERNS, detect_candlesticks
from ..patterns.chart import (
    CHART_PATTERNS,
    detect_chart_patterns,
    support_resistance,
)

# Weight of each chart/candlestick pattern in the composite score.
# Trend signals carry more weight than single candlesticks by design.
_WEIGHTS = {
    "golden_cross": 3.0,
    "death_cross": -3.0,
    "macd_bull_cross": 1.5,
    "macd_bear_cross": -1.5,
    "breakout_20d": 2.5,
    "breakdown_20d": -2.5,
    "rsi_oversold_turn": 1.5,
    "rsi_overbought_turn": -1.5,
    "near_52w_high": 1.0,
    "near_52w_low": -1.0,
    "bb_lower_tag": 0.5,
    "bb_upper_tag": -0.5,
    # candlesticks (lighter; they only confirm)
    "bullish_engulfing": 1.0,
    "bearish_engulfing": -1.0,
    "hammer": 0.8,
    "shooting_star": -0.8,
    "hanging_man": -0.6,
    "inverted_hammer": 0.4,
    "piercing_line": 0.8,
    "dark_cloud_cover": -0.8,
    "morning_star": 1.2,
    "evening_star": -1.2,
    "doji": 0.0,
}

_LABELS = {k: lbl for k, lbl, _ in CHART_PATTERNS + CANDLESTICK_PATTERNS}


@dataclass
class Signal:
    """Scored signal for one symbol as of the latest bar."""

    symbol: str
    date: pd.Timestamp
    close: float
    score: float
    verdict: str  # "Strong Buy" / "Buy" / "Neutral" / "Sell" / "Strong Sell"
    reasons: list[str] = field(default_factory=list)
    indicators: dict = field(default_factory=dict)
    levels: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        """Flat dict for tabular display / DataFrame construction."""
        return {
            "symbol": self.symbol,
            "date": self.date.date() if hasattr(self.date, "date") else self.date,
            "close": round(self.close, 2),
            "score": round(self.score, 2),
            "verdict": self.verdict,
            "rsi": _r(self.indicators.get("rsi_14")),
            "adx": _r(self.indicators.get("adx")),
            "reasons": ", ".join(self.reasons) if self.reasons else "—",
        }


def _r(x, n=1):
    return round(float(x), n) if x is not None and pd.notna(x) else None


def _verdict(score: float) -> str:
    if score >= 4:
        return "Strong Buy"
    if score >= 1.5:
        return "Buy"
    if score <= -4:
        return "Strong Sell"
    if score <= -1.5:
        return "Sell"
    return "Neutral"


def signal_for_frame(symbol: str, df: pd.DataFrame) -> Signal:
    """Compute the latest-bar signal for an already-fetched OHLCV frame."""
    if df.empty:
        raise ValueError(f"no data for {symbol}")

    enriched = add_all(df)
    candles = detect_candlesticks(df)
    charts = detect_chart_patterns(enriched)
    flags = pd.concat([charts, candles], axis=1)

    last = enriched.iloc[-1]
    last_flags = flags.iloc[-1]

    score = 0.0
    reasons: list[str] = []
    for key, fired in last_flags.items():
        if fired and _WEIGHTS.get(key, 0):
            score += _WEIGHTS[key]
            reasons.append(_LABELS.get(key, key))

    # Trend context tilt: price above/below the 200 SMA biases the reading.
    sma200 = last.get("sma_200")
    if pd.notna(sma200):
        if last["close"] > sma200:
            score += 0.5
            reasons.append("Above 200-SMA (uptrend)")
        else:
            score -= 0.5
            reasons.append("Below 200-SMA (downtrend)")

    indicators = {
        "rsi_14": last.get("rsi_14"),
        "macd_hist": last.get("macd_hist"),
        "adx": last.get("adx"),
        "atr_14": last.get("atr_14"),
        "sma_50": last.get("sma_50"),
        "sma_200": last.get("sma_200"),
        "bb_pct_b": last.get("bb_pct_b"),
    }

    return Signal(
        symbol=symbol,
        date=df.index[-1],
        close=float(last["close"]),
        score=score,
        verdict=_verdict(score),
        reasons=reasons,
        indicators=indicators,
        levels=support_resistance(df),
    )


def analyse(
    symbol: str,
    *,
    provider: str = "yfinance",
    provider_config: Optional[dict] = None,
    period_days: int = 400,
    interval: str = "1d",
) -> Signal:
    """Fetch data for ``symbol`` and return its latest-bar :class:`Signal`."""
    prov = get_provider(provider, provider_config)
    df = prov.history(symbol, interval=interval, period_days=period_days)
    return signal_for_frame(symbol, df)
