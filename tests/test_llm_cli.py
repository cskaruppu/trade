"""Tests for the Claude Code (CLI) AI backend — fully mocked, no real CLI."""

import nsetrade.llm_cli as llm_cli
from nsetrade.ai import ThesisConfig, ThesisWriter


def test_mode_resolution_explicit_and_auto(monkeypatch):
    # explicit api needs a key
    assert ThesisConfig(api_key="sk", provider="api").mode == "api"
    assert ThesisConfig(api_key=None, provider="api").mode == "none"

    # explicit claude_cli needs the CLI present
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: True)
    assert ThesisConfig(provider="claude_cli").mode == "claude_cli"
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: False)
    assert ThesisConfig(provider="claude_cli").mode == "none"

    # auto: key wins; else CLI; else none
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: True)
    assert ThesisConfig(api_key="sk").mode == "api"
    assert ThesisConfig(api_key=None).mode == "claude_cli"
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: False)
    assert ThesisConfig(api_key=None).mode == "none"
    assert ThesisConfig(api_key=None).enabled is False


def test_writer_uses_cli_for_text(monkeypatch):
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: True)
    calls = {}

    def fake_run(prompt, *, system=None, model=None, timeout=180):
        calls["prompt"] = prompt
        calls["system"] = system
        calls["model"] = model
        return "BIAS: Bullish\nSETUP: cup base"

    monkeypatch.setattr(llm_cli, "run_claude_cli", fake_run)
    w = ThesisWriter(ThesisConfig(provider="claude_cli", model="claude-opus-4-8"))
    out = w.write("RELIANCE", {"timeframe": "daily",
                               "signal": {"verdict": "Buy", "score": 0.4}})
    assert "Bullish" in out
    assert calls["model"] == "claude-opus-4-8"
    assert "RELIANCE" in calls["prompt"]


def test_classify_headlines_via_cli_parses_json(monkeypatch):
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: True)
    monkeypatch.setattr(
        llm_cli, "run_claude_cli",
        lambda *a, **k: 'Here you go: ["positive", "negative", "neutral"]')
    w = ThesisWriter(ThesisConfig(provider="claude_cli"))
    out = w.classify_headlines(["a", "b", "c"])
    assert out == ["positive", "negative", "neutral"]


def test_classify_headlines_via_cli_pads_and_sanitizes(monkeypatch):
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: True)
    monkeypatch.setattr(llm_cli, "run_claude_cli",
                        lambda *a, **k: '["positive", "weird"]')
    w = ThesisWriter(ThesisConfig(provider="claude_cli"))
    out = w.classify_headlines(["a", "b", "c"])
    assert out == ["positive", "neutral", "neutral"]   # sanitized + padded


def test_vision_requires_api_in_cli_mode(monkeypatch):
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: True)
    w = ThesisWriter(ThesisConfig(provider="claude_cli"))
    try:
        w.read_chart(b"\x89PNG", "RELIANCE")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "vision" in str(exc).lower() or "api key" in str(exc).lower()
