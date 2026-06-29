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


_OPP_SYSTEM = (
    "You are a disciplined technical analyst for Indian (NSE) equities. "
    "You are given a pre-ranked shortlist of trade candidates, each with its "
    "multi-timeframe conviction, the chart pattern present, that pattern's "
    "HISTORICAL win-rate on this stock, and reward:risk. "
    "Give a portfolio-level read: pick the 2-3 highest-quality setups and say "
    "why (lean on the evidence — conviction + pattern edge + R:R), flag any that "
    "look weak despite ranking (e.g. tiny pattern sample), and note what to wait "
    "for before entering. Ground everything in the numbers given — invent "
    "nothing. Be concise (under 200 words). This is educational analysis, not "
    "investment advice, and past pattern stats do not guarantee future results."
)


def build_opportunities_prompt(opportunities, side: str = "long") -> str:
    """Pure prompt builder for the opportunity-shortlist summary."""
    lines = [f"Ranked {side} candidates (best first):"]
    for i, o in enumerate(opportunities, 1):
        row = o.as_row() if hasattr(o, "as_row") else o
        lines.append(
            f"{i}. {row['symbol']}: score {row['score']}, "
            f"conviction {row['conviction']} ({row['aligned']}), "
            f"pattern {row['pattern']} (edge {row['edge']}), "
            f"R:R {row['rr']}, signal {row['verdict']}")
    lines.append("\nGive the portfolio-level read now.")
    return "\n".join(lines)


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
    fib = context.get("fibonacci")
    if fib:
        ext = (f", extension targets {fib['ext_targets']}"
               if fib.get("ext_targets") else "")
        lines.append(
            f"Fibonacci: swing {fib['swing']}; price near the "
            f"{fib['nearest_level']} level @ {fib['nearest_price']} "
            f"(acting as {fib['role']}){ext}")
    plan = context.get("trade_plan")
    if plan:
        lines.append(
            f"Suggested risk frame: entry {plan.get('entry')}, stop {plan.get('stop')}, "
            f"target {plan.get('target')}, R:R {plan.get('rr')}")
    fund = context.get("fundamentals")
    if fund:
        lines.append(f"Fundamentals: {fund}")
    news = context.get("news")
    if news:
        lines.append("Recent headlines: " + " | ".join(news[:4]))
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
        return self._send(_SYSTEM, prompt)

    def summarize_opportunities(self, opportunities, side: str = "long") -> str:
        """Give a portfolio-level read over a ranked opportunity list."""
        return self._send(_OPP_SYSTEM, build_opportunities_prompt(opportunities, side))

    def read_chart(self, image_png: bytes, symbol: str,
                   context: Optional[dict] = None) -> str:
        """Vision: let Claude *look at* the chart image and give an analyst read."""
        import base64

        b64 = base64.standard_b64encode(image_png).decode("ascii")
        text = f"This is the price chart for {symbol} (NSE)."
        if context:
            text += "\n\nComputed context:\n" + build_prompt(symbol, context)
        text += ("\n\nRead the chart like a technical analyst: trend, the chart "
                 "patterns and support/resistance you can SEE, where price is in "
                 "its range, and what would confirm or invalidate a move. Ground "
                 "your read in the image. Be concise (under 180 words). "
                 "Educational only — not investment advice.")
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=1500,
            system=("You are a professional technical analyst for NSE equities "
                    "reading a candlestick chart image."),
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": b64}},
                {"type": "text", "text": text},
            ]}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()

    def _send(self, system: str, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=2000,
            thinking={"type": "adaptive"},
            system=system,
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

    try:
        from .fibonacci import fib_extension, fib_retracement
        fr = fib_retracement(df)
        if fr.found and fr.nearest:
            fx = fib_extension(df)
            ctx["fibonacci"] = {
                "swing": f"{fr.swing_low:.1f}-{fr.swing_high:.1f} ({fr.direction})",
                "nearest_level": fr.nearest.label,
                "nearest_price": round(fr.nearest.price, 2),
                "role": "support" if fr.direction == "up" else "resistance",
                "ext_targets": ([round(l.price, 2) for l in fx.levels
                                 if l.ratio >= 1.0] if fx.found else []),
            }
    except Exception:  # noqa: BLE001 - fib is best-effort enrichment
        pass
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
    if "head" in n and "shoulder" in n:
        return "head_shoulders"
    if "wedge" in n:
        return "wedge"
    if "vcp" in n or "volatility contraction" in n:
        return "vcp"
    return next((k for k in ADVANCED_DETECTORS if k in n), None)
