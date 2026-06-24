"""nsetrade — professional local dashboard.

Run with:
    pip install -e ".[dashboard]"
    streamlit run dashboard/app.py

Tabs: Analyse (interactive candlestick chart + signal + patterns + trade plan),
Watchlist (manage + pattern scan), Confluence (multi-timeframe agreement),
Pattern Edge (historical follow-through), Screener and Backtest.

Binds to localhost only (see .streamlit/config.toml) — private to your machine.
"""

from __future__ import annotations

import streamlit as st

from nsetrade import watchlist as wl
from nsetrade.backtest import backtest_risk
from nsetrade.ai import ThesisConfig, ThesisWriter, assemble_context
from nsetrade.charts_interactive import build_figure
from nsetrade.config import load_config, provider_config
from nsetrade.confluence import confluence_for_frames
from nsetrade.data import get_provider
from nsetrade.edge import all_pattern_edges
from nsetrade.patterns import detect_advanced
from nsetrade.resample import resample_ohlcv, scale_period_days
from nsetrade.screener import screen
from nsetrade.signals.engine import signal_for_frame
from nsetrade.tradeplan import trade_plan
from nsetrade.universe import UNIVERSES, get_universe

st.set_page_config(page_title="nsetrade", page_icon="📈", layout="wide")
cfg = load_config()

# ---- light cosmetic polish -------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.4rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .stTabs [data-baseweb="tab"] {font-size: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _daily(provider, symbol, days):
    prov = get_provider(provider, provider_config(cfg, provider))
    return prov.history(symbol, period_days=days)


def _history(provider, symbol, days, timeframe="daily"):
    df = _daily(provider, symbol, scale_period_days(days, timeframe))
    return resample_ohlcv(df, timeframe)


def _get(sig, key, default=float("nan")):
    val = sig.indicators.get(key)
    return val if val is not None else default


def _verdict_color(v: str) -> str:
    if "Buy" in v:
        return "#26a69a"
    if "Sell" in v:
        return "#ef5350"
    return "#90a4ae"


# ---- sidebar ---------------------------------------------------------------
st.sidebar.title("📈 nsetrade")
st.sidebar.caption("Private NSE analysis — runs only on this machine.")
provider = st.sidebar.selectbox(
    "Data provider", ["yfinance", "kite"],
    index=0 if cfg.get("default_provider") != "kite" else 1,
)
timeframe = st.sidebar.radio("Timeframe", ["daily", "weekly", "monthly"],
                             horizontal=True)
st.sidebar.caption("Kite credentials come from config.yaml. yfinance needs none.")
st.sidebar.warning("Research/education only — not investment advice.")

tab_a, tab_w, tab_c, tab_e, tab_s, tab_b = st.tabs(
    ["📊 Analyse", "⭐ Watchlist", "🎯 Confluence", "🧪 Pattern Edge",
     "🔎 Screener", "📈 Backtest"]
)


# ---- Analyse ---------------------------------------------------------------
with tab_a:
    col1, col2, col3 = st.columns([2, 1, 1])
    symbol = col1.text_input("NSE symbol", value="RELIANCE").strip().upper()
    days = col2.slider("History (days)", 200, 1500, 500, step=50)
    capital = col3.number_input("Capital (₹)", value=100000, step=10000)

    if symbol:
        try:
            df = _history(provider, symbol, days, timeframe)
            sig = signal_for_frame(symbol, df)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Close", f"{sig.close:,.2f}")
            m2.markdown(
                f"**Signal**<br><span style='color:{_verdict_color(sig.verdict)};"
                f"font-size:1.3rem;font-weight:600'>{sig.verdict}</span>"
                f"<br><small>score {sig.score:+.2f}</small>",
                unsafe_allow_html=True)
            m3.metric("RSI(14)", f"{_get(sig, 'rsi_14'):.1f}")
            m4.metric("ADX(14)", f"{_get(sig, 'adx'):.1f}")
            m5.metric("ATR(14)", f"{_get(sig, 'atr_14'):.2f}")

            fig, notes = build_figure(f"{symbol} · {timeframe}", df, bars=220)
            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom": True, "displaylogo": False})

            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Why this signal**")
                st.write(", ".join(sig.reasons) if sig.reasons
                         else "no notable candlestick/indicator patterns")
                st.markdown("**Structural patterns** (heuristic)")
                if notes:
                    for n in notes:
                        st.write(f"• {n}")
                else:
                    st.caption("none detected on this timeframe")
            with cB:
                st.markdown("**Auto trade plan**")
                direction = st.radio("Direction", ["long", "short"],
                                     horizontal=True, key="plan_dir")
                risk = st.slider("Risk per trade %", 0.5, 3.0, 1.0, 0.5,
                                 key="plan_risk") / 100
                pat = next((m for m in detect_advanced(df)
                            if m.breakout_level and m.support and
                            ((direction == "long") == (m.direction == "bullish"))),
                           None)
                try:
                    plan = trade_plan(symbol, df, direction=direction,
                                      capital=float(capital), risk_pct=risk,
                                      pattern=pat)
                    st.code(plan.describe())
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"plan unavailable: {exc}")

            # ---- AI thesis (opt-in; the one feature that leaves the machine) ----
            st.divider()
            tcfg = ThesisConfig.from_config(cfg)
            if not tcfg.enabled:
                st.caption("💡 Add an Anthropic API key (ai.api_key in config.yaml "
                           "or ANTHROPIC_API_KEY) to get an AI-written trade thesis "
                           "grounded in this analysis.")
            elif st.button("🤖 Write AI trade thesis", key="ai_thesis"):
                with st.spinner(f"Asking Claude ({tcfg.model})…"):
                    try:
                        frames = {t: _history(provider, symbol, days, t)
                                  for t in ("daily", "weekly", "monthly")}
                        ctx = assemble_context(symbol, df, timeframe=timeframe,
                                               with_confluence_frames=frames,
                                               capital=float(capital))
                        thesis = ThesisWriter(tcfg).write(symbol, ctx)
                        st.markdown(thesis)
                        st.caption("AI-generated, grounded in the numbers above. "
                                   "Educational only — not investment advice.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"thesis failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not analyse {symbol}: {exc}")


# ---- Watchlist -------------------------------------------------------------
with tab_w:
    st.subheader("Your watchlist")
    st.caption("Stored locally in watchlist.txt (git-ignored, stays private).")
    current = wl.load()
    colA, colB = st.columns([3, 1])
    add_text = colA.text_input("Add symbols (comma/space separated)",
                               placeholder="RELIANCE, INFY, TCS")
    if colB.button("Add", use_container_width=True) and add_text:
        wl.add(add_text.replace(",", " ").split())
        st.rerun()
    if current:
        drop = st.multiselect("Remove symbols", current)
        if st.button("Remove selected") and drop:
            wl.remove(drop)
            st.rerun()
        st.write(f"**{len(current)} symbols:** " + ", ".join(current))
    else:
        st.info("Empty. Add above, or import an NSE CSV via the CLI: "
                "`nsetrade watchlist import EQUITY_L.csv`.")

    st.divider()
    st.markdown("**Scan the watchlist for chart patterns**")
    scan_tfs = st.multiselect("Timeframes", ["daily", "weekly", "monthly"],
                              default=["daily", "weekly"])
    breakouts_only = st.checkbox("Breakouts only (hide still-forming)")
    if st.button("Scan patterns", type="primary") and current:
        rows = []
        prog = st.progress(0.0)
        total = len(current) * max(len(scan_tfs), 1)
        done = 0
        for sym in current:
            for tf in scan_tfs:
                done += 1
                prog.progress(done / total, text=f"{sym} ({tf})")
                try:
                    d = _history(provider, sym, 500, tf)
                    for m in detect_advanced(d):
                        if breakouts_only and m.status != "breakout":
                            continue
                        rows.append({"symbol": sym, "tf": tf, "pattern": m.name,
                                     "direction": m.direction, "status": m.status,
                                     "breakout": m.breakout_level, "note": m.note})
                except Exception:  # noqa: BLE001
                    continue
        prog.empty()
        st.dataframe(rows, use_container_width=True, hide_index=True) if rows \
            else st.warning("No patterns detected.")


# ---- Confluence ------------------------------------------------------------
with tab_c:
    st.subheader("Multi-timeframe confluence")
    st.caption("Stocks where daily, weekly and monthly signals agree score "
               "highest — higher conviction than a single timeframe.")
    src = st.radio("Source", ["My watchlist", "Universe"], horizontal=True,
                   key="conf_src")
    uni = st.selectbox("Universe", list(UNIVERSES), index=0,
                       disabled=(src != "Universe"), key="conf_uni")
    if st.button("Run confluence scan", type="primary"):
        syms = wl.load() if src == "My watchlist" else get_universe(uni)
        if not syms:
            st.warning("No symbols.")
        else:
            prog = st.progress(0.0)
            rows = []
            for i, sym in enumerate(syms):
                prog.progress((i + 1) / len(syms), text=sym)
                try:
                    frames = {tf: _history(provider, sym, 400, tf)
                              for tf in ("daily", "weekly", "monthly")}
                    rows.append(confluence_for_frames(sym, frames).as_row())
                except Exception:  # noqa: BLE001
                    continue
            prog.empty()
            if rows:
                rows.sort(key=lambda r: r["conviction"], reverse=True)
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.warning("No results.")


# ---- Pattern Edge ----------------------------------------------------------
with tab_e:
    st.subheader("Pattern edge — does it historically work?")
    st.caption("For each structural pattern, how past breakouts on this stock "
               "performed afterwards. Small samples are noisy — read the count.")
    e1, e2, e3 = st.columns(3)
    esym = e1.text_input("Symbol", value="RELIANCE", key="edge_sym").strip().upper()
    fwd = e2.slider("Forward bars", 5, 60, 20)
    tgt = e3.slider("Target move %", 2, 20, 5) / 100
    if st.button("Measure edge", type="primary"):
        try:
            df = _history(provider, esym, 1500, timeframe)
            edges = [e for e in all_pattern_edges(df, forward_bars=fwd, target_pct=tgt)
                     if e.occurrences > 0]
            if edges:
                edges.sort(key=lambda e: e.occurrences, reverse=True)
                st.dataframe([{
                    "pattern": e.pattern, "breakouts": e.occurrences,
                    "win rate": f"{e.win_rate:.0%}",
                    "avg return": f"{e.avg_return:+.1%}",
                    f">{tgt:.0%} move": f"{e.hit_target_rate:.0%}",
                    "avg max gain": f"{e.avg_max_favorable:+.1%}",
                    "avg max dd": f"{e.avg_max_adverse:+.1%}",
                } for e in edges], use_container_width=True, hide_index=True)
            else:
                st.info("No historical breakouts of these patterns on this symbol.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not measure edge: {exc}")


# ---- Screener --------------------------------------------------------------
with tab_s:
    st.subheader("Screen & rank")
    ssrc = st.radio("Source", ["Universe", "My watchlist"], horizontal=True,
                    key="scr_src")
    suni = st.selectbox("Universe", list(UNIVERSES), index=0,
                        disabled=(ssrc != "Universe"), key="scr_uni")
    top = st.slider("Top N per side", 5, 30, 10)
    if st.button("Run screen", type="primary"):
        symbols = wl.load() if ssrc == "My watchlist" else get_universe(suni)
        if not symbols:
            st.warning("No symbols.")
        else:
            prog = st.progress(0.0, text="scanning…")
            result = screen(
                symbols, provider=provider,
                provider_config=provider_config(cfg, provider),
                timeframe=timeframe,
                on_progress=lambda d, t, s: prog.progress(d / t, text=f"{s} ({d}/{t})"),
            )
            prog.empty()
            df = result.to_frame()
            if df.empty:
                st.warning("No results (data errors). Try the yfinance provider.")
            else:
                st.markdown("**Top bullish**")
                st.dataframe(df.head(top), use_container_width=True, hide_index=True)
                st.markdown("**Top bearish**")
                st.dataframe(df.tail(top).iloc[::-1], use_container_width=True,
                             hide_index=True)


# ---- Backtest --------------------------------------------------------------
with tab_b:
    st.subheader("Risk-managed backtest")
    b1, b2, b3 = st.columns(3)
    bsym = b1.text_input("Symbol", value="RELIANCE", key="bt_sym").strip().upper()
    strat = b2.selectbox("Strategy",
                         ["rsi_ma", "sma_crossover", "macd", "breakout"])
    years = b3.slider("Years", 1, 8, 3)
    r1, r2, r3 = st.columns(3)
    stop_atr = r1.slider("Stop (ATR mult)", 0.5, 5.0, 2.0, step=0.5)
    target_atr = r2.slider("Target (ATR mult, 0=off)", 0.0, 10.0, 4.0, step=0.5)
    risk = r3.slider("Risk per trade %", 0.5, 5.0, 1.0, step=0.5) / 100
    if st.button("Run backtest", type="primary"):
        try:
            df = _history(provider, bsym, int(years * 365) + 30, timeframe)
            res = backtest_risk(bsym, df, strategy=strat, stop_atr=stop_atr,
                                target_atr=target_atr or None, risk_per_trade=risk)
            m = res.metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total return", f"{m['total_return']:+.1%}")
            k2.metric("CAGR", f"{m['cagr']:+.1%}")
            k3.metric("Sharpe", f"{m['sharpe']:.2f}")
            k4.metric("Max drawdown", f"{m['max_drawdown']:.1%}")
            st.line_chart(res.equity_curve, height=280)
            st.code(res.summary())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest failed: {exc}")
