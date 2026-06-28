"""Tests for the natural-language screener and the chart-vision AI read."""

import json

from nsetrade import nlscreen
from nsetrade.ai import ThesisConfig
from nsetrade.opportunities import Opportunity


def _opp(symbol, **kw):
    base = dict(score=3.0, side="long", close=100.0, conviction=2.0,
                aligned="bullish", signal_verdict="Buy", pattern="Cup & Handle",
                pattern_win_rate=0.6, pattern_occurrences=6, rr=2.0, rsi=55.0)
    base.update(kw)
    return Opportunity(symbol=symbol, **base)


OPPS = [
    _opp("A", score=4.0, conviction=3.0, rr=2.5, pattern_win_rate=0.7, rsi=60),
    _opp("B", score=1.0, conviction=0.5, rr=1.2, pattern_win_rate=0.4, rsi=80,
         aligned="mixed", signal_verdict="Neutral", pattern=None),
    _opp("C", score=2.5, conviction=2.0, rr=3.0, pattern="Double Bottom",
         pattern_win_rate=0.55, rsi=40),
]


def test_normalize_spec_fills_defaults():
    s = nlscreen.normalize_spec({"min_rr": 2.0})
    assert s["min_rr"] == 2.0
    assert s["side"] == "long"
    assert s["verdict"] == "any"


def test_apply_filter_min_rr_and_pattern():
    s = {"min_rr": 2.0, "require_pattern": True}
    out = [o.symbol for o in nlscreen.apply_filter(OPPS, s)]
    assert "A" in out and "C" in out
    assert "B" not in out          # rr 1.2 and no pattern


def test_apply_filter_pattern_contains():
    out = [o.symbol for o in nlscreen.apply_filter(OPPS, {"pattern_contains": "cup"})]
    assert out == ["A"]


def test_apply_filter_rsi_and_winrate():
    out = [o.symbol for o in nlscreen.apply_filter(
        OPPS, {"max_rsi": 65, "min_win_rate": 0.65})]
    assert out == ["A"]          # B rsi 80 excluded; C win 0.55 excluded


def test_apply_filter_aligned_and_verdict():
    out = [o.symbol for o in nlscreen.apply_filter(
        OPPS, {"aligned": "bullish", "verdict": "Buy"})]
    assert set(out) == {"A", "C"}     # B is mixed/Neutral


def test_build_query_prompt():
    p = nlscreen.build_query_prompt("bullish cups with good edge")
    assert "bullish cups" in p


class _Block:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _FakeMessages:
    def __init__(self, payload, rec):
        self._payload = payload
        self._rec = rec

    def create(self, **kwargs):
        self._rec.append(kwargs)
        return type("R", (), {"content": [_Block(self._payload)]})()


class _FakeClient:
    def __init__(self, payload):
        self.calls = []
        self.messages = _FakeMessages(payload, self.calls)


def test_nlscreener_parse_returns_normalized_spec():
    payload = json.dumps({
        "side": "long", "min_score": -999, "min_conviction": 2.0,
        "verdict": "Buy", "aligned": "bullish", "require_pattern": True,
        "pattern_contains": "cup", "min_win_rate": 0.6, "min_rr": 2.0,
        "min_rsi": -1, "max_rsi": 70, "explanation": "bullish cups with edge"})
    client = _FakeClient(payload)
    scr = nlscreen.NLScreener(ThesisConfig(api_key="sk-test"), client=client)
    spec = scr.parse("bullish cups with a real edge and 2:1")
    assert spec["pattern_contains"] == "cup"
    assert spec["min_rr"] == 2.0
    # the request asked for constrained JSON output
    assert client.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_vision_read_chart_sends_image_block():
    from nsetrade.ai import ThesisWriter
    client = _FakeClient("Uptrend intact; cup forming; watch 105 breakout.")
    w = ThesisWriter(ThesisConfig(api_key="sk-test"), client=client)
    out = w.read_chart(b"\x89PNG_fake_bytes", "TCS")
    assert "Uptrend" in out
    content = client.calls[0]["messages"][0]["content"]
    kinds = [b["type"] for b in content]
    assert "image" in kinds and "text" in kinds
    img = next(b for b in content if b["type"] == "image")
    assert img["source"]["media_type"] == "image/png"
