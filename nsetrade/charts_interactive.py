"""Interactive Plotly candlestick charts with indicator and pattern overlays.

Builds a polished, dark-themed multi-pane figure (price + volume + RSI + MACD)
and draws detected structural patterns — breakout level, support and the
pattern's box/range — directly on the price pane. Returns a Plotly Figure so the
Streamlit dashboard can render it interactively (zoom, hover, pan).

Install with ``pip install -e ".[dashboard]"``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .indicators import add_all
from .patterns.advanced import detect_advanced

# A restrained, professional dark palette.
_BG = "#0e1117"
_GRID = "#1f2530"
_UP = "#26a69a"
_DOWN = "#ef5350"
_MA = {"sma_20": "#42a5f5", "sma_50": "#ffa726", "sma_200": "#ab47bc"}


def build_figure(
    symbol: str,
    df: pd.DataFrame,
    *,
    bars: int = 200,
    show_patterns: bool = True,
    show_fib: bool = False,
):
    """Return a Plotly Figure for ``df`` (last ``bars`` rows)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    enriched = add_all(df)
    view = enriched.tail(bars)
    x = view.index

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.15, 0.15], vertical_spacing=0.03,
        subplot_titles=("", "Volume", "RSI (14)", "MACD"),
    )

    # ---- price candles ----
    fig.add_trace(go.Candlestick(
        x=x, open=view["open"], high=view["high"], low=view["low"],
        close=view["close"], name="Price",
        increasing_line_color=_UP, decreasing_line_color=_DOWN,
        increasing_fillcolor=_UP, decreasing_fillcolor=_DOWN,
    ), row=1, col=1)

    for col, color in _MA.items():
        if view[col].notna().any():
            fig.add_trace(go.Scatter(
                x=x, y=view[col], name=col.upper().replace("_", " "),
                line=dict(color=color, width=1.2)), row=1, col=1)

    # Bollinger band as a shaded envelope
    if view["bb_upper"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=view["bb_upper"], name="BB",
                                 line=dict(color="#546e7a", width=0.6),
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=view["bb_lower"], name="BB",
                                 line=dict(color="#546e7a", width=0.6),
                                 fill="tonexty", fillcolor="rgba(84,110,122,0.10)",
                                 showlegend=False), row=1, col=1)

    # ---- pattern overlays ----
    pattern_notes = []
    if show_patterns:
        for m in detect_advanced(df):
            pattern_notes.append(m.describe())
            color = "#26a69a" if m.direction == "bullish" else "#ef5350"
            if m.breakout_level:
                fig.add_hline(y=m.breakout_level, line=dict(color=color, width=1,
                              dash="dash"), row=1, col=1,
                              annotation_text=f"{m.name} breakout {m.breakout_level:.1f}",
                              annotation_position="top left",
                              annotation_font_color=color)
            if m.support and m.start is not None:
                # shade the pattern's range from its start to the last bar
                try:
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=m.start, x1=view.index[-1],
                        y0=m.support, y1=m.breakout_level or m.resistance or m.support,
                        line=dict(width=0), fillcolor="rgba(120,144,156,0.08)",
                        row=1, col=1)
                except Exception:  # noqa: BLE001 - overlay is best-effort
                    pass
            # draw the pattern's actual shape (cup U-curve, W, neckline, …)
            for ov in (getattr(m, "overlays", None) or []):
                try:
                    fig.add_trace(go.Scatter(
                        x=ov["x"], y=ov["y"], mode="lines",
                        line=dict(color="#e6e9ef", width=2,
                                  shape="spline" if ov.get("kind") == "spline"
                                  else "linear"),
                        name=m.name, showlegend=False, hoverinfo="skip"),
                        row=1, col=1)
                except Exception:  # noqa: BLE001 - overlay is best-effort
                    pass

    # ---- Fibonacci retracement overlay ----
    if show_fib:
        try:
            from .fibonacci import fib_retracement
            fr = fib_retracement(df)
            if fr.found:
                for lvl in fr.levels:
                    fig.add_hline(
                        y=lvl.price, line=dict(color="#c9a227", width=0.7,
                                               dash="dot"), row=1, col=1,
                        annotation_text=f"fib {lvl.label} {lvl.price:.1f}",
                        annotation_position="right",
                        annotation_font_color="#c9a227",
                        annotation_font_size=9)
        except Exception:  # noqa: BLE001 - overlay is best-effort
            pass

    # ---- volume ----
    vol_colors = [_UP if c >= o else _DOWN
                  for c, o in zip(view["close"], view["open"])]
    fig.add_trace(go.Bar(x=x, y=view["volume"], marker_color=vol_colors,
                         name="Volume", showlegend=False), row=2, col=1)

    # ---- RSI ----
    fig.add_trace(go.Scatter(x=x, y=view["rsi_14"], line=dict(color="#ce93d8"),
                             name="RSI", showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line=dict(color=_DOWN, width=0.6, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=_UP, width=0.6, dash="dot"), row=3, col=1)

    # ---- MACD ----
    fig.add_trace(go.Bar(x=x, y=view["macd_hist"], name="Hist",
                         marker_color="#607d8b", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=view["macd"], line=dict(color="#42a5f5"),
                             name="MACD", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=view["macd_signal"], line=dict(color="#ffa726"),
                             name="Signal", showlegend=False), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        title=dict(text=f"{symbol}", x=0.01, font=dict(size=20)),
        height=760, margin=dict(l=40, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        dragmode="pan",
    )
    for r in range(1, 5):
        fig.update_xaxes(gridcolor=_GRID, row=r, col=1)
        fig.update_yaxes(gridcolor=_GRID, row=r, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig, pattern_notes
