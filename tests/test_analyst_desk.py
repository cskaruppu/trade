"""Tests for the AI analyst desk — aggregation, grading, veto — with a fake client."""

import json

import pytest

from nsetrade import analyst_desk as desk
from nsetrade.ai import ThesisConfig


CTX = {
    "timeframe": "daily",
    "signal": {"verdict": "Buy", "score": 2.5, "close": 100.0, "rsi": 58.0,
               "adx": 30.0, "reasons": ["Golden Cross (50/200)"]},
    "levels": {"support": [90.0], "resistance": [110.0]},
    "confluence": {"conviction": 3.1, "aligned": "bullish",
                   "per_tf": {"daily": "Buy", "weekly": "Buy"}},
    "patterns": [{"name": "Cup & Handle", "direction": "bullish",
                  "status": "breakout", "breakout_level": 105.0,
                  "edge": {"occurrences": 6, "win_rate": "67%", "avg_return": "+4%"}}],
    "trade_plan": {"entry": 100.0, "stop": 95.0, "target": 110.0, "rr": 2.0},
}


class _Block:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _Messages:
    """Returns a per-role score based on a {role_key: score} map, matched via
    the role's angle text appearing in the system prompt."""

    def __init__(self, scores, recorder):
        self._scores = scores
        self._rec = recorder

    def create(self, **kwargs):
        self._rec.append(kwargs)
        system = kwargs["system"]
        score = 70
        for role in desk.ROLES:
            if role.angle[:25] in system:
                score = self._scores.get(role.key, 70)
                break
        payload = {"score": score, "stance": "ok", "concerns": ["c1"]}
        return type("R", (), {"content": [_Block(json.dumps(payload))]})()


class _Client:
    def __init__(self, scores):
        self.calls = []
        self.messages = _Messages(scores, self.calls)


def _desk(scores):
    return desk.AnalystDesk(ThesisConfig(api_key="sk-test"), client=_Client(scores))


def test_build_agent_prompt_has_context_and_role():
    p = desk.build_agent_prompt(desk.ROLES[0], "TCS", CTX)
    assert "TCS" in p
    assert desk.ROLES[0].title in p
    assert "Cup & Handle" in p


def test_high_scores_give_grade_a():
    g = _desk({"trend": 90, "evidence": 88, "risk": 85, "skeptic": 82}).grade(
        "TCS", CTX, side="long")
    assert g.grade == "A"
    assert g.composite >= 80
    assert len(g.verdicts) == 4
    assert "Grade A" in g.describe()


def test_low_scores_give_grade_f():
    g = _desk({"trend": 30, "evidence": 25, "risk": 20, "skeptic": 15}).grade(
        "X", CTX)
    assert g.grade == "F"


def test_skeptic_veto_caps_top_grade():
    # everyone loves it but the skeptic shreds it → capped at C, not A/B
    g = _desk({"trend": 95, "evidence": 95, "risk": 95, "skeptic": 10}).grade(
        "X", CTX)
    assert g.grade == "C"  # veto applied


def test_makes_one_call_per_role():
    d = _desk({"trend": 70, "evidence": 70, "risk": 70, "skeptic": 70})
    d.grade("X", CTX)
    assert len(d._client.calls) == len(desk.ROLES)
    # each call requested structured JSON output
    for c in d._client.calls:
        assert c["output_config"]["format"]["type"] == "json_schema"


def test_scores_are_clamped_to_0_100():
    d = _desk({"trend": 999, "evidence": -50, "risk": 70, "skeptic": 70})
    g = d.grade("X", CTX)
    by = {v.role: v.score for v in g.verdicts}
    assert by["trend"] == 100
    assert by["evidence"] == 0


def test_letter_thresholds():
    assert desk._letter(85, 80) == "A"
    assert desk._letter(72, 80) == "B"
    assert desk._letter(64, 80) == "C"
    assert desk._letter(54, 80) == "D"
    assert desk._letter(40, 80) == "F"
    assert desk._letter(85, 20) == "C"  # veto
