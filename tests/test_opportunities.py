"""Tests for the opportunity ranker — pure scoring + ranking with a fake provider."""

import numpy as np
import pandas as pd

from nsetrade import opportunities as opp


def ohlcv(close, start="2019-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(np.asarray(close, float), index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.01
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.99
    v = pd.Series(np.full(len(close), 1e5), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def test_score_opportunity_uptrend_scores_positive():
    df = ohlcv(np.linspace(80, 220, 600) + np.sin(np.arange(600) / 15) * 6)
    o = opp.score_opportunity("UP", df, side="long", with_edge=True)
    assert o.symbol == "UP"
    assert o.score > 0  # a clean uptrend should be a positive long opportunity
    assert o.conviction > 0
    assert o.close > 0
    # result is explainable: components are populated
    row = o.as_row()
    assert set(["symbol", "score", "conviction", "pattern", "edge", "rr"]) <= row.keys()


def test_downtrend_long_score_below_uptrend():
    # realistic (noisy) trends — a flat linspace is a degenerate signal input
    wob = np.sin(np.arange(600) / 14) * 6
    up = ohlcv(np.linspace(80, 220, 600) + wob)
    down = ohlcv(np.linspace(220, 80, 600) + wob)
    o_up = opp.score_opportunity("UP", up, side="long")
    o_down = opp.score_opportunity("DOWN", down, side="long")
    assert o_up.score > o_down.score


class _FakeProvider:
    """Serves canned frames; lets us test ranking without network."""

    def __init__(self, frames):
        self._frames = frames

    def history(self, symbol, **kwargs):
        if symbol not in self._frames:
            raise ValueError(f"no data for {symbol}")
        return self._frames[symbol]


def test_rank_orders_by_score_and_collects_errors():
    frames = {
        "STRONG": ohlcv(np.linspace(80, 240, 600)),
        "WEAK": ohlcv(np.linspace(150, 120, 600)),
    }
    prov = _FakeProvider(frames)
    res = opp.rank_opportunities(
        ["STRONG", "WEAK", "MISSING"], side="long", top=15,
        _provider_obj=prov)
    syms = [o.symbol for o in res.opportunities]
    assert syms[0] == "STRONG"          # best setup ranked first
    assert "MISSING" in res.errors      # per-symbol error captured, not fatal
    assert "STRONG" in res.to_frame()["symbol"].values


def test_top_limits_results():
    frames = {f"S{i}": ohlcv(np.linspace(80, 80 + i * 20, 600)) for i in range(5)}
    res = opp.rank_opportunities(list(frames), top=2, _provider_obj=_FakeProvider(frames))
    assert len(res.opportunities) == 2


def test_min_score_filters():
    frames = {"A": ohlcv(np.linspace(220, 80, 600))}  # downtrend → low long score
    res = opp.rank_opportunities(["A"], side="long", min_score=999,
                                 _provider_obj=_FakeProvider(frames))
    assert res.opportunities == []


def test_build_opportunities_prompt():
    from nsetrade.ai import build_opportunities_prompt
    o = opp.Opportunity(symbol="TCS", score=4.2, side="long", close=3900.0,
                        conviction=3.1, aligned="bullish", signal_verdict="Buy",
                        pattern="Cup & Handle", pattern_win_rate=0.67,
                        pattern_occurrences=6, rr=2.0)
    prompt = build_opportunities_prompt([o], side="long")
    assert "TCS" in prompt
    assert "Cup & Handle" in prompt
    assert "67%/6" in prompt
    assert prompt.strip().endswith("Give the portfolio-level read now.")
