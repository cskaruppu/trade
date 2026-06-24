"""Smoke test for the interactive Plotly chart builder (no browser needed)."""

import numpy as np
import pandas as pd


def _df(n=300):
    idx = pd.bdate_range("2022-01-01", periods=n)
    p = np.concatenate([np.linspace(100, 80, n // 3),
                        np.linspace(80, 160, n - n // 3)])
    c = pd.Series(p, index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    h = pd.concat([o, c], axis=1).max(axis=1) * 1.01
    l = pd.concat([o, c], axis=1).min(axis=1) * 0.99
    v = pd.Series(np.random.RandomState(0).randint(2e5, 9e5, n), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def test_build_figure():
    from nsetrade.charts_interactive import build_figure

    fig, notes = build_figure("TEST", _df(), bars=200)
    # candlestick + MAs + bollinger + volume + rsi + macd traces present
    assert len(fig.data) >= 6
    assert isinstance(notes, list)
    # serialises to HTML without needing a browser/kaleido
    html = fig.to_html(include_plotlyjs=False)
    assert len(html) > 1000
