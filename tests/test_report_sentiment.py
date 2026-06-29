"""Tests for the Markdown report builder and AI headline-sentiment tagging."""

import json

from nsetrade.ai import ThesisConfig, ThesisWriter
from nsetrade.report import build_report_md


def test_build_report_md_full():
    md = build_report_md(
        "RELIANCE", "weekly", pattern="Cup & Handle",
        confidence="High confidence", edge="70% / 8",
        entry="100.00", stop="95.00", target="115.00", rr="3.0:1", size=200,
        debt_status="Debt-free ✓", highlights=["Sector: Energy", "P/E: 24.5"],
        news=[{"title": "Q3 beat", "publisher": "ET", "sentiment": "positive"},
              {"title": "Capex concern", "publisher": "Mint", "sentiment": "negative"}],
        thesis="Bullish: cup breakout with proven edge.")
    assert "# RELIANCE — EdgeForge report" in md
    assert "Cup & Handle" in md and "High confidence" in md
    assert "Entry: 100.00" in md and "Stop loss: 95.00" in md
    assert "Debt-free ✓" in md
    assert "🟢 Q3 beat" in md and "🔴 Capex concern" in md
    assert "## AI thesis" in md
    assert "not investment advice" in md


def test_build_report_md_minimal():
    md = build_report_md("X", "daily")
    assert "# X — EdgeForge report" in md
    assert "## Pattern" not in md       # nothing supplied
    assert "## Trade plan" not in md


class _Block:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    class _M:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.calls.append(kwargs)
            return type("R", (), {"content": [_Block(self.outer._payload)]})()

    @property
    def messages(self):
        return _FakeClient._M(self)


def test_classify_headlines_aligns_and_pads():
    client = _FakeClient(json.dumps({"sentiments": ["positive", "negative"]}))
    w = ThesisWriter(ThesisConfig(api_key="sk-test"), client=client)
    out = w.classify_headlines(["good news", "bad news", "third one"])
    assert out == ["positive", "negative", "neutral"]   # padded to 3
    # used structured JSON output
    assert client.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_classify_headlines_empty():
    w = ThesisWriter(ThesisConfig(api_key="sk-test"), client=_FakeClient("{}"))
    assert w.classify_headlines([]) == []


def test_classify_headlines_sanitizes_bad_labels():
    client = _FakeClient(json.dumps({"sentiments": ["POSITIVE", "weird", "neutral"]}))
    w = ThesisWriter(ThesisConfig(api_key="sk-test"), client=client)
    out = w.classify_headlines(["a", "b", "c"])
    assert out == ["positive", "neutral", "neutral"]   # POSITIVE lowercased, weird→neutral
