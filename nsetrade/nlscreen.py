"""Natural-language screener — describe what you want, Claude builds the filter.

Type "bullish weekly setups near a cup & handle with a strong historical edge
and at least 2:1 reward" and Claude translates it into a structured filter spec
(via constrained JSON output). We then rank the universe with the normal
engine and apply the filter in plain Python — so the *matching* is
deterministic and auditable; only the natural-language → spec step uses the LLM.

The filter spec uses sentinel "unset" values (-999 / -1 / "" / "any") so the
model can always return every field and we never depend on nullable schemas.
"""

from __future__ import annotations

import json
from typing import Optional

# Structured-output schema. Every field is required; "unset" sentinels keep the
# model from having to emit nulls.
FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "side": {"type": "string", "enum": ["long", "short"]},
        "min_score": {"type": "number"},          # -999 = unset
        "min_conviction": {"type": "number"},     # -999 = unset
        "verdict": {"type": "string", "enum": ["Buy", "Sell", "any"]},
        "aligned": {"type": "string", "enum": ["bullish", "bearish", "any"]},
        "require_pattern": {"type": "boolean"},
        "pattern_contains": {"type": "string"},    # "" = unset (e.g. "cup")
        "min_win_rate": {"type": "number"},        # -1 = unset (0..1)
        "min_rr": {"type": "number"},              # -1 = unset
        "min_rsi": {"type": "number"},             # -1 = unset
        "max_rsi": {"type": "number"},             # -1 = unset
        "explanation": {"type": "string"},
    },
    "required": ["side", "min_score", "min_conviction", "verdict", "aligned",
                 "require_pattern", "pattern_contains", "min_win_rate",
                 "min_rr", "min_rsi", "max_rsi", "explanation"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You convert a retail trader's plain-English stock-screening request into a "
    "filter spec for an NSE technical screener. Use ONLY the provided fields. "
    "For anything the user didn't ask for, use the 'unset' sentinel: -999 for "
    "min_score/min_conviction, -1 for min_win_rate/min_rr/min_rsi/max_rsi, "
    "\"\" for pattern_contains, \"any\" for verdict/aligned, false for "
    "require_pattern. win_rate is a fraction 0..1. RSI is 0..100. Put a short "
    "plain-English read of how you interpreted the request in 'explanation'. "
    "Return the JSON only."
)

# defaults applied to any missing key so apply_filter is robust to partial specs
_DEFAULTS = {
    "side": "long", "min_score": -999, "min_conviction": -999, "verdict": "any",
    "aligned": "any", "require_pattern": False, "pattern_contains": "",
    "min_win_rate": -1, "min_rr": -1, "min_rsi": -1, "max_rsi": -1,
    "explanation": "",
}


def build_query_prompt(query: str) -> str:
    """Pure prompt builder — wraps the user's request."""
    return (f"Screening request: {query!r}\n\n"
            "Produce the filter spec JSON.")


def normalize_spec(spec: dict) -> dict:
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in (spec or {}).items() if k in _DEFAULTS})
    return out


def apply_filter(opportunities, spec: dict) -> list:
    """Filter ranked :class:`Opportunity` objects by a (normalized) spec.

    Pure and deterministic — no LLM, no network. This is the auditable half of
    the natural-language screener.
    """
    s = normalize_spec(spec)
    out = []
    for o in opportunities:
        if s["min_score"] > -900 and o.score < s["min_score"]:
            continue
        if s["min_conviction"] > -900 and o.conviction < s["min_conviction"]:
            continue
        if s["verdict"] == "Buy" and "Buy" not in (o.signal_verdict or ""):
            continue
        if s["verdict"] == "Sell" and "Sell" not in (o.signal_verdict or ""):
            continue
        if s["aligned"] != "any" and o.aligned != s["aligned"]:
            continue
        if s["require_pattern"] and not o.pattern:
            continue
        pc = s["pattern_contains"].strip().lower()
        if pc and (not o.pattern or pc not in o.pattern.lower()):
            continue
        if s["min_win_rate"] >= 0 and (o.pattern_win_rate is None
                                       or o.pattern_win_rate < s["min_win_rate"]):
            continue
        if s["min_rr"] >= 0 and (o.rr is None or o.rr < s["min_rr"]):
            continue
        if s["min_rsi"] >= 0 and (o.rsi is None or o.rsi < s["min_rsi"]):
            continue
        if s["max_rsi"] >= 0 and (o.rsi is None or o.rsi > s["max_rsi"]):
            continue
        out.append(o)
    return out


class NLScreener:
    """Turns a plain-English request into a filter spec via Claude."""

    def __init__(self, config, client=None):
        self.config = config
        if client is not None:
            self._client = client
        else:
            from .ai import ThesisWriter
            self._client = ThesisWriter(config)._client

    def parse(self, query: str) -> dict:
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=700,
            system=_SYSTEM,
            messages=[{"role": "user", "content": build_query_prompt(query)}],
            output_config={"format": {"type": "json_schema",
                                      "schema": FILTER_SCHEMA}},
        )
        text = next((b.text for b in resp.content
                     if getattr(b, "type", None) == "text"), "{}")
        return normalize_spec(json.loads(text))

    def screen(self, query: str, symbols: list[str], *,
               provider: str = "yfinance", provider_config: Optional[dict] = None,
               period_days: int = 500, top: int = 25, on_progress=None):
        """Parse the query, rank the universe, and return (spec, matches)."""
        from .opportunities import rank_opportunities
        spec = self.parse(query)
        ranked = rank_opportunities(
            symbols, provider=provider, provider_config=provider_config,
            period_days=period_days, side=spec["side"], top=0,
            on_progress=on_progress)
        matches = apply_filter(ranked.opportunities, spec)[:top]
        return spec, matches
