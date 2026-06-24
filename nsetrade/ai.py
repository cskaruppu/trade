"""AI-written trade thesis using Claude (the Anthropic API).

This is the one feature that reaches outside your machine: it sends a compact,
**numeric summary** of the analysis (signal, indicators, detected patterns and
their historical edge, multi-timeframe confluence) to Anthropic's API and gets
back a short, structured trade thesis. It is strictly opt-in — it only runs when
you provide an Anthropic API key — and it never sends raw price data, only the
derived summary.

Prompt assembly (:func:`build_prompt`) is a pure function and unit-tested; the
network call is isolated in :class:`ThesisWriter` so it can be mocked.

Requires ``pip install -e ".[ai]"`` and an API key (``ai.api_key`` in
config.yaml or the ``ANTHROPIC_API_KEY`` environment variable).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Claude model + system prompt. Opus 4.8 is the current most-capable model.
DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You are a disciplined technical analyst for Indian (NSE) equities. "
    "You are given a numeric summary of one stock's current technical state. "
    "Write a concise, structured trade thesis a retail swing trader can act on. "
    "Be specific and grounded ONLY in the numbers provided — do not invent "
    "fundamentals, news, or price levels that aren't given. Always include risk. "
    "This is educational analysis, not investment advice.\n\n"
    "Format your answer as:\n"
    "BIAS: <Bullish/Bearish/Neutral> (one line why)\n"
    "SETUP: <the key pattern/signal confluence in 1-2 sentences>\n"
    "EVIDENCE: <cite the pattern edge stats / multi-timeframe agreement>\n"
    "PLAN: <entry trigger, stop, target idea — reference the given levels>\n"
    "RISK: <what would invalidate this; note small-sample caveats>\n"
    "Keep it under 180 words. No hype."
)


@dataclass
class ThesisConfig:
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL

    @classmethod
    def from_config(cls, cfg: dict) -> "ThesisConfig":
        a = (cfg or {}).get("ai", {}) or {}
        return cls(
            api_key=a.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"),
            model=a.get("model", DEFAULT_MODEL),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def build_prompt(symbol: str, context: dict) -> str:
    """Turn a structured analysis ``context`` dict into the user prompt text.

    Pure function — no network, no SDK. ``context`` is assembled by
    :func:`assemble_context` but any dict with the same shape works.
    """
    lines = [f"Stock: {symbol} (NSE)", f"Timeframe: {context.get('timeframe', 'daily')}"]
    sig = context.get("signal") or {}
    if sig:
        lines.append(
            f"Signal: {sig.get('verdict')} (score {sig.get('score')}); "
            f"RSI {sig.get('rsi')}, ADX {sig.get('adx')}, "
            f"close {sig.get('close')}"
        )
        if sig.get("reasons"):
            lines.append("Active signals: " + ", ".join(sig["reasons"]))
    lv = context.get("levels") or {}
    if lv:
        lines.append(f"Support: {lv.get('support')}  Resistance: {lv.get('resistance')}")
    conf = context.get("confluence")
    if conf:
        lines.append(
            f"Multi-timeframe: conviction {conf.get('conviction')}, "
            f"aligned={conf.get('aligned')} "
            f"({', '.join(f'{k}:{v}' for k, v in (conf.get('per_tf') or {}).items())})"
        )
    patterns = context.get("patterns") or []
    if patterns:
        lines.append("Detected chart patterns:")
        for p in patterns:
            edge = p.get("edge")
            edge_txt = ""
            if edge and edge.get("occurrences"):
                edge_txt = (f"  [history: {edge['occurrences']} past breakouts, "
                            f"{edge['win_rate']} positive, avg {edge['avg_return']}]")
            lines.append(
                f"  - {p.get('name')} ({p.get('direction')}/{p.get('status')}), "
                f"breakout {p.get('breakout_level')}{edge_txt}")
    plan = context.get("trade_plan")
    if plan:
        lines.append(
            f"Suggested risk frame: entry {plan.get('entry')}, stop {plan.get('stop')}, "
            f"target {plan.get('target')}, R:R {plan.get('rr')}")
    lines.append("\nWrite the trade thesis now.")
    return "\n".join(lines)


class ThesisWriter:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, config: ThesisConfig, client=None):
        self.config = config
        if client is not None:
            self._client = client
            return
        if not config.enabled:
            raise ValueError(
                "AI thesis needs an Anthropic API key. Set ai.api_key in "
                "config.yaml or the ANTHROPIC_API_KEY environment variable."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "anthropic is not installed. Run: pip install -e \".[ai]\""
            ) from exc
        self._client = anthropic.Anthropic(api_key=config.api_key)

    def write(self, symbol: str, context: dict) -> str:
        """Generate a trade thesis. Returns the model's text."""
        prompt = build_prompt(symbol, context)
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=2000,
            thinking={"type": "adaptive"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content is a list of blocks; collect the text blocks only
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()


def assemble_context(
    symbol: str,
    df,
    *,
    timeframe: str = "daily",
    with_confluence_frames: Optional[dict] = None,
    with_edge: bool = True,
    capital: float = 100_000.0,
    risk_pct: float = 0.01,
) -> dict:
    """Build the analysis context dict the thesis is grounded in.

    Runs the signal engine, structural pattern detectors (optionally with their
    historical edge), a trade plan, and — if daily/weekly/monthly frames are
    supplied — the multi-timeframe confluence.
    """
    from .signals.engine import signal_for_frame
    from .patterns import detect_advanced
    from .tradeplan import trade_plan

    sig = signal_for_frame(symbol, df)
    ctx = {
        "timeframe": timeframe,
        "signal": {
            "verdict": sig.verdict,
            "score": round(sig.score, 2),
            "close": round(sig.close, 2),
            "rsi": _r(sig.indicators.get("rsi_14")),
            "adx": _r(sig.indicators.get("adx")),
            "reasons": sig.reasons,
        },
        "levels": sig.levels,
    }

    patterns = []
    for m in detect_advanced(df):
        entry = {"name": m.name, "direction": m.direction, "status": m.status,
                 "breakout_level": _r(m.breakout_level, 2)}
        if with_edge:
            try:
                from .edge import pattern_edge
                key = _pattern_key(m.name)
                if key:
                    e = pattern_edge(df, key, forward_bars=20)
                    if e.occurrences:
                        entry["edge"] = {
                            "occurrences": e.occurrences,
                            "win_rate": f"{e.win_rate:.0%}",
                            "avg_return": f"{e.avg_return:+.1%}",
                        }
            except Exception:  # noqa: BLE001 - edge is best-effort enrichment
                pass
        patterns.append(entry)
    ctx["patterns"] = patterns

    try:
        direction = "long" if "Buy" in sig.verdict or sig.score >= 0 else "short"
        plan = trade_plan(symbol, df, direction=direction, capital=capital,
                          risk_pct=risk_pct)
        ctx["trade_plan"] = {"entry": plan.entry, "stop": plan.stop,
                             "target": plan.target, "rr": plan.rr}
    except Exception:  # noqa: BLE001
        pass

    if with_confluence_frames:
        from .confluence import confluence_for_frames
        c = confluence_for_frames(symbol, with_confluence_frames)
        ctx["confluence"] = {
            "conviction": round(c.conviction, 2),
            "aligned": c.aligned,
            "per_tf": {tf: s.verdict for tf, s in c.per_timeframe.items()},
        }
    return ctx


def _r(x, n=1):
    try:
        return round(float(x), n) if x is not None and x == x else None
    except (TypeError, ValueError):
        return None


def _pattern_key(name: str) -> Optional[str]:
    """Map a PatternMatch display name back to its detector key."""
    from .patterns.advanced import ADVANCED_DETECTORS
    n = name.lower()
    if "cup" in n:
        return "cup_and_handle"
    if "darvas" in n:
        return "darvas_box"
    if "flag" in n:
        return "flag"
    if "double bottom" in n:
        return "double_bottom"
    if "double top" in n:
        return "double_top"
    if "triangle" in n:
        return "triangle"
    return next((k for k in ADVANCED_DETECTORS if k in n), None)
