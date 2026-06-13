"""Render annotated price charts to PNG (matplotlib).

Produces a multi-panel chart — price with moving averages and Bollinger bands,
volume, RSI and MACD — and marks the most recent detected candlestick/chart
patterns directly on the price panel. Designed to be glanceable on a phone.

Uses the non-interactive Agg backend so it works headless (no display needed).
Install with ``pip install -e ".[charts]"``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .indicators import add_all
from .patterns.candlestick import CANDLESTICK_PATTERNS, detect_candlesticks
from .patterns.chart import CHART_PATTERNS, detect_chart_patterns

_BIAS = {k: b for k, _, b in CHART_PATTERNS + CANDLESTICK_PATTERNS}
_LABEL = {k: lbl for k, lbl, _ in CHART_PATTERNS + CANDLESTICK_PATTERNS}


def render_chart(
    symbol: str,
    df: pd.DataFrame,
    *,
    out_path: Optional[str] = None,
    bars: int = 180,
    mark_last: int = 30,
) -> str:
    """Render ``df`` to a PNG and return the output path.

    Parameters
    ----------
    bars:       number of most-recent bars to plot.
    mark_last:  annotate pattern hits only within this many trailing bars
                (keeps the chart readable).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    enriched = add_all(df)
    candles = detect_candlesticks(df)
    charts = detect_chart_patterns(enriched)
    flags = pd.concat([charts, candles], axis=1)

    view = enriched.tail(bars)
    flags = flags.loc[view.index]
    x = view.index

    fig, (ax_p, ax_v, ax_r, ax_m) = plt.subplots(
        4, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )
    fig.suptitle(f"{symbol} — daily", fontsize=14, fontweight="bold")

    # ---- price panel: close + MAs + Bollinger ----
    ax_p.plot(x, view["close"], color="black", linewidth=1.2, label="Close")
    for col, color in [("sma_20", "tab:blue"), ("sma_50", "tab:orange"),
                       ("sma_200", "tab:red")]:
        if view[col].notna().any():
            ax_p.plot(x, view[col], color=color, linewidth=0.9, label=col.upper())
    if view["bb_upper"].notna().any():
        ax_p.fill_between(x, view["bb_lower"], view["bb_upper"],
                          color="tab:gray", alpha=0.12, label="Bollinger")
    ax_p.set_ylabel("Price")
    ax_p.legend(loc="upper left", fontsize=8, ncol=3)
    ax_p.grid(alpha=0.2)

    # ---- annotate recent pattern hits ----
    recent = view.tail(mark_last)
    seen = set()
    for key in _LABEL:
        if key not in flags.columns:
            continue
        hits = flags[key].loc[recent.index]
        for dt in hits[hits].index:
            bias = _BIAS.get(key, 0)
            color = "green" if bias > 0 else "red" if bias < 0 else "gray"
            marker = "^" if bias > 0 else "v" if bias < 0 else "o"
            price = view.loc[dt, "low"] if bias > 0 else view.loc[dt, "high"]
            offset = -1 if bias > 0 else 1
            ax_p.scatter([dt], [price], marker=marker, color=color,
                         s=40, zorder=5)
            label = _LABEL[key]
            if label not in seen:  # avoid annotation spam
                ax_p.annotate(label, (dt, price),
                              textcoords="offset points",
                              xytext=(0, offset * 12), ha="center",
                              fontsize=7, color=color)
                seen.add(label)

    # ---- volume ----
    colors = ["tab:green" if c >= o else "tab:red"
              for c, o in zip(view["close"], view["open"])]
    ax_v.bar(x, view["volume"], color=colors, width=1.0)
    if view["vol_sma_20"].notna().any():
        ax_v.plot(x, view["vol_sma_20"], color="black", linewidth=0.8)
    ax_v.set_ylabel("Volume")
    ax_v.grid(alpha=0.2)

    # ---- RSI ----
    ax_r.plot(x, view["rsi_14"], color="tab:purple", linewidth=1.0)
    ax_r.axhline(70, color="red", linewidth=0.7, linestyle="--")
    ax_r.axhline(30, color="green", linewidth=0.7, linestyle="--")
    ax_r.set_ylim(0, 100)
    ax_r.set_ylabel("RSI")
    ax_r.grid(alpha=0.2)

    # ---- MACD ----
    ax_m.plot(x, view["macd"], color="tab:blue", linewidth=1.0, label="MACD")
    ax_m.plot(x, view["macd_signal"], color="tab:orange", linewidth=1.0,
              label="Signal")
    ax_m.bar(x, view["macd_hist"], color="tab:gray", width=1.0, alpha=0.5)
    ax_m.axhline(0, color="black", linewidth=0.6)
    ax_m.set_ylabel("MACD")
    ax_m.legend(loc="upper left", fontsize=7)
    ax_m.grid(alpha=0.2)

    ax_m.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_m.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = out_path or f"{symbol}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
