"""AI analyst desk — a panel of specialized agents that grade a setup.

One AI opinion is a guess. A *panel* that scrutinizes a candidate from
independent angles — and includes a skeptic whose job is to kill the trade —
produces a far more trustworthy verdict. Each agent returns a 0-100 score with
its stance and concerns; the desk aggregates them into a letter grade (A-F).

This is the honest version of "high accuracy": it doesn't predict price, it
raises the **precision of the shortlist** — only setups that survive the whole
panel earn an A. A strong skeptic argument vetoes a top grade.

Cost note: every agent is a separate Claude API call, so run the desk only on
the *top-ranked* candidates from the fast local screen — never the whole
universe. Prompt assembly is pure/testable; the API calls are isolated and
mockable (inject a client).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

# Each agent: a focused persona + how much its vote counts. The skeptic and the
# evidence verifier carry the most weight — evidence and disconfirmation matter
# most for trustworthiness.
_BASE = (
    "You are one member of a technical-analysis review panel for an NSE stock. "
    "You are given a numeric summary of the setup. Score it 0-100 for how good "
    "a {side} trade it is FROM YOUR SPECIFIC ANGLE ONLY, grounded strictly in "
    "the numbers given — invent nothing. Higher = more favourable. "
    "Respond with JSON: score (0-100 integer), stance (one sentence), concerns "
    "(array of short strings). This is educational analysis, not advice."
)


@dataclass(frozen=True)
class AgentRole:
    key: str
    title: str
    weight: float
    angle: str  # appended to the base system prompt

    def system(self, side: str) -> str:
        return _BASE.format(side=side) + "\n\nYOUR ANGLE: " + self.angle


ROLES: list[AgentRole] = [
    AgentRole("trend", "Trend analyst", 1.0,
              "Judge trend quality and multi-timeframe agreement (conviction, "
              "alignment, signal score). Is the broader trend really behind this?"),
    AgentRole("evidence", "Pattern-edge verifier", 1.3,
              "Judge the chart pattern and its HISTORICAL edge — especially "
              "whether it held up out-of-sample and whether the sample is large "
              "enough to trust. A tiny or fading edge should score low."),
    AgentRole("risk", "Risk manager", 1.0,
              "Judge reward:risk, the stop distance, and downside. A poor R:R or "
              "an entry far from support should score low regardless of trend."),
    AgentRole("skeptic", "Devil's advocate", 1.3,
              "Your job is to KILL this trade. Argue the bear case (or bull case "
              "for a short). Score LOW if there are real reasons it fails; only "
              "score high if the setup genuinely survives hard scrutiny."),
]

_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "stance": {"type": "string"},
        "concerns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "stance", "concerns"],
    "additionalProperties": False,
}


@dataclass
class AgentVerdict:
    role: str
    title: str
    score: int
    stance: str
    concerns: list[str] = field(default_factory=list)


@dataclass
class DeskGrade:
    symbol: str
    side: str
    grade: str            # A | B | C | D | F
    composite: float      # 0-100 weighted score
    verdicts: list[AgentVerdict]
    summary: str

    def describe(self) -> str:
        lines = [f"{self.symbol} ({self.side}) — Grade {self.grade} "
                 f"(panel score {self.composite:.0f}/100)"]
        for v in self.verdicts:
            lines.append(f"  {v.title:<22} {v.score:>3}/100 — {v.stance}")
        if self.summary:
            lines.append(f"  → {self.summary}")
        return "\n".join(lines)


def build_agent_prompt(role: AgentRole, symbol: str, context: dict) -> str:
    """Pure prompt builder for one panel agent (re-uses the thesis context)."""
    from .ai import build_prompt
    return (f"Panel role: {role.title}\n\n"
            + build_prompt(symbol, context)
            + "\n\nScore this setup from your angle and return the JSON.")


def _letter(score: float, skeptic: Optional[int]) -> str:
    """Map the composite to a letter, with a skeptic veto on the top grades."""
    if score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 50:
        grade = "D"
    else:
        grade = "F"
    # a strong kill argument caps the grade — the panel must not over-rate a
    # setup the skeptic shredded
    if skeptic is not None and skeptic < 30 and grade in ("A", "B"):
        grade = "C"
    return grade


class AnalystDesk:
    """Runs the agent panel against a candidate context."""

    def __init__(self, config, client=None):
        # config is an ai.ThesisConfig (reuses the same API key/model wiring)
        self.config = config
        if client is not None:
            self._client = client
        else:
            from .ai import ThesisWriter
            # borrow ThesisWriter's client construction (key + SDK guards)
            self._client = ThesisWriter(config)._client

    def _judge(self, role: AgentRole, symbol: str, context: dict,
               side: str) -> AgentVerdict:
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=1000,
            system=role.system(side),
            messages=[{"role": "user",
                       "content": build_agent_prompt(role, symbol, context)}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        text = next((b.text for b in resp.content
                     if getattr(b, "type", None) == "text"), "{}")
        data = json.loads(text)
        score = max(0, min(100, int(data.get("score", 0))))
        return AgentVerdict(role=role.key, title=role.title, score=score,
                            stance=str(data.get("stance", "")).strip(),
                            concerns=[str(c) for c in data.get("concerns", [])])

    def grade(self, symbol: str, context: dict, *, side: str = "long",
              roles: Optional[list[AgentRole]] = None) -> DeskGrade:
        roles = roles or ROLES
        verdicts = [self._judge(r, symbol, context, side) for r in roles]
        wsum = sum(r.weight for r in roles)
        composite = sum(v.score * r.weight for v, r in zip(verdicts, roles)) / wsum
        skeptic = next((v.score for v in verdicts if v.role == "skeptic"), None)
        grade = _letter(composite, skeptic)
        summary = _summarize(grade, verdicts)
        return DeskGrade(symbol=symbol, side=side, grade=grade,
                         composite=composite, verdicts=verdicts, summary=summary)


def _summarize(grade: str, verdicts: list[AgentVerdict]) -> str:
    low = min(verdicts, key=lambda v: v.score)
    high = max(verdicts, key=lambda v: v.score)
    if grade in ("A", "B"):
        return (f"Panel is constructive; strongest read from {high.title.lower()}. "
                f"Watch: {low.title.lower()} ({low.score}/100).")
    if grade == "C":
        return f"Mixed — {low.title.lower()} is the main drag ({low.score}/100)."
    return f"Panel is unconvinced; {low.title.lower()} scored {low.score}/100."
