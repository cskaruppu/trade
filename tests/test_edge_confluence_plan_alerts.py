"""Tests for pattern-edge, confluence, trade-plan and alert modules."""

import numpy as np
import pandas as pd

from nsetrade import alerts, confluence, edge, tradeplan
from nsetrade.resample import resample_ohlcv


def ohlcv(close, start="2018-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(np.asarray(close, float), index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.005
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.995
    v = pd.Series(np.full(len(close), 1e5), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def _staircase(cycles=4, box_len=45, ramp=6, seed=0):
    """Repeated tight-box → breakout staircases (lots of Darvas breakouts)."""
    rng = np.random.RandomState(seed)
    base = 100.0
    segs = []
    for _ in range(cycles):
        segs.append(base + rng.uniform(-3, 3, box_len))     # tight box
        segs.append(np.linspace(base, base + 18, ramp))     # breakout ramp
        base += 18
    return ohlcv(np.concatenate(segs))


# ---- pattern edge --------------------------------------------------------

def test_pattern_edge_structure():
    df = _staircase()
    e = edge.pattern_edge(df, "darvas_box", forward_bars=10)
    assert 0.0 <= e.win_rate <= 1.0
    assert 0.0 <= e.hit_target_rate <= 1.0
    assert e.forward_bars == 10
    assert "darvas_box" in e.describe()


def test_pattern_edge_finds_breakouts():
    df = _staircase(cycles=5)
    e = edge.pattern_edge(df, "darvas_box", forward_bars=8)
    assert e.occurrences >= 1
    assert len(e.samples) == e.occurrences


def test_pattern_edge_unknown_raises():
    df = ohlcv(np.linspace(100, 110, 120))
    try:
        edge.pattern_edge(df, "nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_all_pattern_edges_runs():
    df = _staircase()
    out = edge.all_pattern_edges(df, forward_bars=10)
    assert len(out) == len(edge.ADVANCED_DETECTORS)


# ---- confluence ----------------------------------------------------------

def test_confluence_for_frames():
    daily = ohlcv(np.linspace(80, 200, 900) + np.sin(np.arange(900) / 20) * 6)
    frames = {tf: resample_ohlcv(daily, tf) for tf in ("daily", "weekly", "monthly")}
    c = confluence.confluence_for_frames("TEST", frames)
    assert c.aligned in {"bullish", "bearish", "mixed"}
    assert c.verdict in {"Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"}
    assert set(c.per_timeframe).issubset({"daily", "weekly", "monthly"})
    assert isinstance(c.conviction, float)
    row = c.as_row()
    assert row["symbol"] == "TEST"


def test_confluence_uptrend_is_bullish():
    # strong steady uptrend should read bullish across timeframes
    daily = ohlcv(np.linspace(50, 250, 900))
    frames = {tf: resample_ohlcv(daily, tf) for tf in ("daily", "weekly", "monthly")}
    c = confluence.confluence_for_frames("UP", frames)
    assert c.conviction > 0


# ---- trade plan ----------------------------------------------------------

def test_trade_plan_long():
    df = ohlcv(np.linspace(100, 150, 120))
    plan = tradeplan.trade_plan("TEST", df, capital=100_000, risk_pct=0.01,
                                stop_atr=2.0, rr_target=2.0)
    assert plan.stop < plan.entry < plan.target
    assert plan.shares > 0
    assert plan.rr > 0
    # risk on the trade should be near the budget (<= a bit over due to rounding)
    assert plan.risk_amount <= 100_000 * 0.01 * 1.2
    assert "Trade plan" in plan.describe()


def test_trade_plan_short():
    df = ohlcv(np.linspace(150, 100, 120))
    plan = tradeplan.trade_plan("TEST", df, direction="short", stop_atr=2.0)
    assert plan.target < plan.entry < plan.stop


def test_trade_plan_pattern_target():
    from nsetrade.patterns.advanced import PatternMatch
    df = ohlcv(np.linspace(100, 150, 120))   # entry ~150
    pm = PatternMatch(name="Darvas Box", found=True, direction="bullish",
                      status="breakout", breakout_level=152.0, support=140.0)
    plan = tradeplan.trade_plan("TEST", df, pattern=pm)
    # measured move target = breakout + (breakout - support) = 164 (beyond entry)
    assert plan.target > plan.entry
    assert "measured move" in plan.rationale


def test_trade_plan_falls_back_when_target_behind_entry():
    from nsetrade.patterns.advanced import PatternMatch
    df = ohlcv(np.linspace(100, 150, 120))   # entry ~150
    # price already ran past the breakout level -> measured move would be behind
    pm = PatternMatch(name="Darvas Box", found=True, direction="bullish",
                      status="breakout", breakout_level=120.0, support=110.0)
    plan = tradeplan.trade_plan("TEST", df, pattern=pm, rr_target=2.0)
    assert plan.target > plan.entry          # never behind entry
    assert "target" in plan.rationale        # used the RR fallback


# ---- alerts --------------------------------------------------------------

def test_format_alert():
    msg = alerts.format_alert("Breakout!", ["RELIANCE Buy", "score +3"])
    assert "Breakout!" in msg and "RELIANCE" in msg


def test_dispatch_console_and_log(tmp_path, capsys):
    log = tmp_path / "alerts.log"
    cfg = alerts.AlertConfig(console=True, log_file=str(log))
    status = alerts.dispatch("hello alert", cfg)
    assert status["console"] and status["log"]
    assert log.exists() and "hello alert" in log.read_text()
    assert "telegram" not in status  # not configured


def test_alert_config_from_config_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    cfg = alerts.AlertConfig.from_config({})
    assert cfg.telegram_enabled
