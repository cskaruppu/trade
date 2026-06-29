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
    show_trendlines: bool = False,
    only_keys=None,
):
    """Return a Plotly Figure for ``df`` (last ``bars`` rows).

    ``only_keys`` (a set of detector keys, e.g. ``{"cup_and_handle"}``) restricts
    the drawn patterns/target to just those — so the chart focuses on the
    pattern the user selected instead of every detected shape.
    """
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
        _matches = detect_advanced(df)
        if only_keys:
            from .ai import _pattern_key
            _matches = [m for m in _matches if _pattern_key(m.name) in only_keys]
        for m in _matches:
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
            # draw the pattern's actual shape (cup U-curve, W, neckline, band…)
            for ov in (getattr(m, "overlays", None) or []):
                try:
                    if ov.get("kind") == "band":      # accumulation/support zone
                        fig.add_hrect(y0=ov["y0"], y1=ov["y1"], line_width=0,
                                      fillcolor="rgba(66,165,245,0.16)",
                                      row=1, col=1)
                        continue
                    fig.add_trace(go.Scatter(
                        x=ov["x"], y=ov["y"], mode="lines",
                        line=dict(color="#e6e9ef", width=2,
                                  shape="spline" if ov.get("kind") == "spline"
                                  else "linear"),
                        name=m.name, showlegend=False, hoverinfo="skip"),
                        row=1, col=1)
                except Exception:  # noqa: BLE001 - overlay is best-effort
                    pass

        # ---- measured-move target ("upside potential") ----
        _bull = [m for m in _matches
                 if m.direction == "bullish" and m.breakout_level and m.support
                 and m.breakout_level > m.support]
        if _bull:
            # prefer a confirmed breakout, then the tallest base
            m = max(_bull, key=lambda x: (x.status == "breakout",
                                          x.breakout_level - x.support))
            entry = float(m.breakout_level)
            target = entry + (entry - float(m.support))   # classic measured move
            last_close = float(view["close"].iloc[-1])
            pct = (target / last_close - 1.0) if last_close else 0.0
            if target > last_close:
                try:
                    fig.add_hrect(y0=entry, y1=target, line_width=0,
                                  fillcolor="rgba(38,166,154,0.10)", row=1, col=1)
                    fig.add_hline(
                        y=target, line=dict(color="#26a69a", width=1.2, dash="dot"),
                        row=1, col=1,
                        annotation_text=f"🎯 Target {target:.1f}  (+{pct:.0%})",
                        annotation_position="top left",
                        annotation_font_color="#26a69a")
                    x_arrow = view.index[int(len(view) * 0.93)]
                    fig.add_annotation(
                        x=x_arrow, y=target, ax=x_arrow, ay=entry,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1.4, arrowwidth=2.5,
                        arrowcolor="#26a69a", text="", row=1, col=1)
                    pattern_notes.append(
                        f"Upside potential: {entry:.1f} → {target:.1f} (+{pct:.0%})")
                except Exception:  # noqa: BLE001 - target is best-effort
                    pass

    # ---- support / resistance trendlines ----
    if show_trendlines:
        try:
            from .trendlines import fit_trendlines
            tl = fit_trendlines(view)
            for kind, color in (("resistance", "#ef9a9a"), ("support", "#80cbc4")):
                ln = tl.get(kind)
                if ln:
                    fig.add_trace(go.Scatter(
                        x=ln["x"], y=ln["y"], mode="lines",
                        line=dict(color=color, width=1.5, dash="dot"),
                        name=kind, showlegend=False, hoverinfo="skip"),
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

    # ---- RSI ----  (the "14" in the title is the period; the value is the line)
    fig.add_trace(go.Scatter(x=x, y=view["rsi_14"], line=dict(color="#ce93d8"),
                             name="RSI", showlegend=False), row=3, col=1)
    if view["rsi_14"].notna().any():
        _last_rsi = float(view["rsi_14"].dropna().iloc[-1])
        _rcol = ("#ef5350" if _last_rsi >= 70 else
                 "#26a69a" if _last_rsi <= 30 else "#ce93d8")
        fig.add_annotation(
            x=x[-1], y=_last_rsi, text=f" {_last_rsi:.0f}", row=3, col=1,
            showarrow=False, xanchor="left", font=dict(color=_rcol, size=11),
            bgcolor="rgba(17,21,28,0.85)")
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
