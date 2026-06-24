"""Tests for the AI thesis module — prompt building + a mocked API client."""

import numpy as np
import pandas as pd

from nsetrade import ai


def ohlcv(close, start="2019-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(np.asarray(close, float), index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.01
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.99
    v = pd.Series(np.full(len(close), 1e5), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def test_thesis_config_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = ai.ThesisConfig.from_config({})
    assert cfg.enabled and cfg.model == ai.DEFAULT_MODEL


def test_thesis_config_disabled(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = ai.ThesisConfig.from_config({})
    assert not cfg.enabled


def test_build_prompt_includes_key_facts():
    ctx = {
        "timeframe": "weekly",
        "signal": {"verdict": "Buy", "score": 2.5, "close": 100.0,
                   "rsi": 58.0, "adx": 30.0, "reasons": ["Golden Cross (50/200)"]},
        "levels": {"support": [90.0], "resistance": [110.0]},
        "confluence": {"conviction": 3.1, "aligned": "bullish",
                       "per_tf": {"daily": "Buy", "weekly": "Buy"}},
        "patterns": [{"name": "Cup & Handle", "direction": "bullish",
                      "status": "breakout", "breakout_level": 105.0,
                      "edge": {"occurrences": 6, "win_rate": "67%",
                               "avg_return": "+4.2%"}}],
        "trade_plan": {"entry": 100.0, "stop": 95.0, "target": 110.0, "rr": 2.0},
    }
    prompt = ai.build_prompt("RELIANCE", ctx)
    assert "RELIANCE" in prompt
    assert "weekly" in prompt
    assert "Golden Cross" in prompt
    assert "Cup & Handle" in prompt
    assert "6 past breakouts" in prompt
    assert "conviction 3.1" in prompt
    assert prompt.strip().endswith("Write the trade thesis now.")


class _FakeBlock:
    def __init__(self, text, type="text"):
        self.text = text
        self.type = type


class _FakeMessages:
    def __init__(self, recorder):
        self._rec = recorder

    def create(self, **kwargs):
        self._rec.update(kwargs)
        # mimic adaptive-thinking response: a thinking block then a text block
        return type("Resp", (), {"content": [
            _FakeBlock("", "thinking"),
            _FakeBlock("BIAS: Bullish\nSETUP: cup breakout\nRISK: small sample"),
        ]})()


class _FakeClient:
    def __init__(self):
        self.calls = {}
        self.messages = _FakeMessages(self.calls)


def test_thesis_writer_with_mock_client():
    client = _FakeClient()
    cfg = ai.ThesisConfig(api_key="sk-test")
    writer = ai.ThesisWriter(cfg, client=client)
    out = writer.write("TEST", {"timeframe": "daily",
                                "signal": {"verdict": "Buy"}})
    # only the text block is returned, thinking is dropped
    assert "BIAS: Bullish" in out
    assert "thinking" not in out
    # correct model + adaptive thinking + no sampling params were sent
    assert client.calls["model"] == ai.DEFAULT_MODEL
    assert client.calls["thinking"] == {"type": "adaptive"}
    assert "temperature" not in client.calls
    assert "budget_tokens" not in str(client.calls.get("thinking"))


def test_assemble_context_shape():
    df = ohlcv(np.linspace(80, 200, 600) + np.sin(np.arange(600) / 15) * 8)
    ctx = ai.assemble_context("TEST", df, timeframe="daily", with_edge=True)
    assert ctx["timeframe"] == "daily"
    assert "verdict" in ctx["signal"]
    assert isinstance(ctx["patterns"], list)
    # prompt builds from the assembled context without error
    prompt = ai.build_prompt("TEST", ctx)
    assert "TEST" in prompt


def test_pattern_key_mapping():
    assert ai._pattern_key("Cup & Handle") == "cup_and_handle"
    assert ai._pattern_key("Darvas Box") == "darvas_box"
    assert ai._pattern_key("Ascending Triangle") == "triangle"
