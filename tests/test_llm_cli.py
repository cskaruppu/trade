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


def test_chat_via_cli_flattens_history_and_grounds(monkeypatch):
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: True)
    seen = {}

    def fake_run(prompt, *, system=None, model=None, timeout=180):
        seen["prompt"] = prompt
        seen["system"] = system
        return "Looks like a tight base; watch the breakout. Risk: a close below support."

    monkeypatch.setattr(llm_cli, "run_claude_cli", fake_run)
    w = ThesisWriter(ThesisConfig(provider="claude_cli"))
    history = [{"role": "user", "content": "Is RELIANCE a buy?"},
               {"role": "assistant", "content": "It depends on the setup."},
               {"role": "user", "content": "What's the risk?"}]
    out = w.chat(history, grounding="Support: 1200  Resistance: 1300")
    assert "Risk" in out
    # history flattened into the single CLI prompt; grounding injected into system
    assert "RELIANCE" in seen["prompt"] and "What's the risk?" in seen["prompt"]
    assert "Support: 1200" in seen["system"]


def test_chat_via_api_passes_messages(monkeypatch):
    class _Block:
        type = "text"
        text = "Grounded answer with risk noted."

    class _Resp:
        content = [_Block()]

    class _Client:
        def __init__(self):
            self.kwargs = None

        class messages:  # noqa: N801
            pass

    client = _Client()
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _Resp()

    client.messages.create = create
    w = ThesisWriter(ThesisConfig(api_key="sk-test"), client=client)
    out = w.chat([{"role": "user", "content": "hi"}], grounding="CTX")
    assert "risk" in out.lower()
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert "CTX" in captured["system"]


def test_vision_requires_api_in_cli_mode(monkeypatch):
    monkeypatch.setattr(llm_cli, "claude_cli_available", lambda: True)
    w = ThesisWriter(ThesisConfig(provider="claude_cli"))
    try:
        w.read_chart(b"\x89PNG", "RELIANCE")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "vision" in str(exc).lower() or "api key" in str(exc).lower()
