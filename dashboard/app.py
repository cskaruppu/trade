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

try:  # added later; fall back gracefully if nsetrade/ is an older checkout
    from nsetrade.universe import list_universes
except ImportError:
    def list_universes():
        return list(UNIVERSES)

st.set_page_config(page_title="nsetrade", page_icon="📈", layout="wide")
cfg = load_config()

# ---- professional polish ---------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 700;}
      [data-testid="stMetricLabel"] {opacity: 0.75;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
      .stTabs [data-baseweb="tab-list"] {gap: 4px;}
      /* hero header banner */
      .nt-hero {
        background: linear-gradient(110deg, #0f2d2a 0%, #11151c 55%, #1a1430 100%);
        border: 1px solid #1f2a33; border-radius: 14px;
        padding: 18px 22px; margin-bottom: 14px;
      }
      .nt-hero h1 {margin: 0; font-size: 1.7rem; color: #e8f3f1;}
      .nt-hero .tag {color: #79c7bd; font-size: 0.95rem; margin-top: 2px;}
      .nt-hero .sub {color: #8b97a3; font-size: 0.82rem; margin-top: 6px;}
      /* grade badges */
      .nt-grade {display:inline-block; min-width: 2.1em; text-align:center;
        font-weight: 800; border-radius: 8px; padding: 2px 10px; color: #06120f;}
      .nt-A {background:#26a69a;} .nt-B {background:#7cc47b;}
      .nt-C {background:#e3b341; color:#1a1400;} .nt-D {background:#e8915a; color:#1a0c00;}
      .nt-F {background:#ef5350; color:#1a0000;}
      .nt-pill {display:inline-block; background:#161b24; border:1px solid #28323d;
        border-radius:999px; padding:3px 12px; margin-right:6px; font-size:0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _grade_badge(grade: str) -> str:
    return f'<span class="nt-grade nt-{grade}">{grade}</span>'


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

# ---- hero header + KPI strip ----------------------------------------------
st.markdown(
    """
    <div class="nt-hero">
      <h1>📈 nsetrade</h1>
      <div class="tag">Evidence-based NSE analysis · AI analyst desk · runs 100% on your machine</div>
      <div class="sub">Every signal is backtestable. Patterns carry their real historical edge.
      Nothing leaves this laptop except the optional AI calls you trigger.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _last_scan_label():
    try:
        from datetime import datetime
        from nsetrade.scan_cache import ScanCache
        runs = ScanCache().list_runs(limit=1)
        if runs:
            return datetime.fromtimestamp(runs[0]["created_at"]).strftime("%d %b %H:%M")
    except Exception:  # noqa: BLE001
        pass
    return "—"


_k1, _k2, _k3, _k4 = st.columns(4)
_k1.metric("Universes", len(list_universes()))
_k2.metric("Data provider", provider)
_k3.metric("Timeframe", timeframe)
_k4.metric("Last cached scan", _last_scan_label())

tab_o, tab_q, tab_a, tab_c, tab_e, tab_s, tab_w, tab_b = st.tabs(
    ["🚀 Opportunities", "💬 Ask AI", "📊 Analyse", "🎯 Confluence",
     "🧪 Pattern Edge", "🔎 Screener", "⭐ Watchlist", "📈 Backtest"]
)


# ---- Ask AI (natural-language screener) ------------------------------------
with tab_q:
    st.subheader("Ask in plain English")
    st.caption("Describe the setup you want — Claude builds the filter, then the "
               "engine ranks and matches it. The LLM only writes the filter; the "
               "matching is deterministic and auditable.")
    _tcfg_ask = ThesisConfig.from_config(cfg)
    examples = ("bullish weekly setups near a cup & handle with at least 2:1 "
                "reward and a 60%+ historical edge")
    q = st.text_area("Your request", value="", placeholder=examples, height=80)
    qc1, qc2 = st.columns([2, 1])
    q_uni = qc1.selectbox("Search universe", list_universes(), index=0, key="ask_uni")
    q_top = qc2.slider("Max matches", 5, 40, 20, step=5, key="ask_top")
    if not _tcfg_ask.enabled:
        st.info("Add an Anthropic API key (ai.api_key or ANTHROPIC_API_KEY) to "
                "use the natural-language screener.")
    elif st.button("🔎 Find matches", type="primary", disabled=not q.strip()):
        from nsetrade.nlscreen import NLScreener
        syms = get_universe(q_uni)
        prog = st.progress(0.0)
        try:
            spec, matches = NLScreener(_tcfg_ask).screen(
                q.strip(), syms, provider=provider,
                provider_config=provider_config(cfg, provider), top=q_top,
                on_progress=lambda d, t, s: prog.progress(d / t, text=s))
            prog.empty()
            st.markdown(f"**Claude read your request as:** {spec.get('explanation','')}")
            if matches:
                st.dataframe([o.as_row() for o in matches],
                             use_container_width=True, hide_index=True)
                st.caption(f"{len(matches)} matches · LLM built the filter, "
                           "matching is deterministic. Educational only.")
            else:
                st.warning("No stocks matched that filter — try loosening it.")
        except Exception as exc:  # noqa: BLE001
            prog.empty()
            st.error(f"Ask AI failed: {exc}")


# ---- Top Opportunities -----------------------------------------------------
with tab_o:
    st.subheader("Top trade opportunities")
    st.caption("Ranks each stock by **multi-timeframe conviction + historical "
               "pattern edge + reward:risk** — the highest-conviction, "
               "evidence-based setups. Probability, not a profit guarantee.")
    oc1, oc2, oc3 = st.columns([2, 1, 1])
    o_src = oc1.radio("Source", ["Universe", "My watchlist"], horizontal=True,
                      key="opp_src")
    o_uni = oc1.selectbox("Universe", list_universes(), index=0,
                          disabled=(o_src != "Universe"), key="opp_uni")
    o_side = oc2.radio("Side", ["long", "short"], horizontal=True, key="opp_side")
    o_top = oc3.slider("Show top", 5, 40, 15, step=5, key="opp_top")
    o_edge = oc2.checkbox("Include pattern edge", value=True, key="opp_edge",
                          help="backtests each pattern's history; slower but "
                               "stronger evidence")

    b1, b2 = st.columns(2)
    if b1.button("Find opportunities (live)", type="primary"):
        from nsetrade.opportunities import rank_opportunities
        syms = wl.load() if o_src == "My watchlist" else get_universe(o_uni)
        if not syms:
            st.warning("No symbols to scan.")
        else:
            prog = st.progress(0.0)
            res = rank_opportunities(
                syms, provider=provider,
                provider_config=provider_config(cfg, provider),
                side=o_side, with_edge=o_edge, top=o_top,
                on_progress=lambda d, t, s: prog.progress(d / t, text=s))
            prog.empty()
            st.session_state["opp_result"] = res
            st.session_state.pop("opp_cached_when", None)
    if b2.button("⚡ Load cached scan (instant)",
                 help="reads the latest 'nsetrade precompute' run — set one up "
                      "to run nightly for big universes"):
        from datetime import datetime
        from nsetrade.scan_cache import ScanCache
        hit = ScanCache().latest("opportunities", f"{o_uni}:{o_side}")
        if not hit:
            st.warning("No cached scan for this universe/side yet. Run "
                       "`nsetrade precompute --universe %s --side %s` (or the "
                       "nightly task)." % (o_uni, o_side))
        else:
            st.session_state["opp_cached_rows"] = hit["rows"][:o_top]
            st.session_state["opp_cached_when"] = datetime.fromtimestamp(
                hit["created_at"]).strftime("%Y-%m-%d %H:%M")
            st.session_state.pop("opp_result", None)

    cached_when = st.session_state.get("opp_cached_when")
    res = st.session_state.get("opp_result")
    if cached_when:
        st.info(f"Showing cached scan from **{cached_when}**.")
        st.dataframe(st.session_state["opp_cached_rows"],
                     use_container_width=True, hide_index=True)
        st.caption("score=composite · conviction=multi-timeframe trend · "
                   "edge=pattern win-rate/sample · rr=reward:risk")
    elif res is not None:
        df = res.to_frame()
        if df.empty:
            st.warning("No opportunities found on this universe/side.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("score=composite · conviction=multi-timeframe trend · "
                       "edge=pattern win-rate/sample · rr=reward:risk")

            # market map: opportunity score across the shortlist
            try:
                import plotly.express as px
                fig = px.bar(df.iloc[::-1], x="score", y="symbol",
                             orientation="h", color="score",
                             color_continuous_scale="Tealgrn",
                             title="Opportunity score map")
                fig.update_layout(height=max(220, 26 * len(df)),
                                  template="plotly_dark", margin=dict(l=8, r=8, t=40, b=8),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:  # noqa: BLE001
                pass

            tcfg = ThesisConfig.from_config(cfg)
            if not tcfg.enabled:
                st.caption("💡 Add an Anthropic API key for an AI portfolio read "
                           "and the multi-agent analyst-desk grades.")
            else:
                cda, cdb = st.columns(2)
                if cda.button("🤖 AI portfolio read", key="opp_ai"):
                    with st.spinner(f"Asking Claude ({tcfg.model})…"):
                        try:
                            st.markdown(ThesisWriter(tcfg).summarize_opportunities(
                                res.opportunities,
                                side=res.opportunities[0].side
                                if res.opportunities else "long"))
                            st.caption("AI-generated, grounded above. "
                                       "Educational only — not advice.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"AI read failed: {exc}")
                n_grade = min(3, len(res.opportunities))
                if cdb.button(f"🧠 Grade top {n_grade} with AI analyst desk",
                              key="opp_desk"):
                    from nsetrade.analyst_desk import AnalystDesk
                    from nsetrade.ai import assemble_context
                    desk = AnalystDesk(tcfg)
                    with st.spinner("Convening the analyst panel "
                                    "(several Claude calls per stock)…"):
                        for o in res.opportunities[:n_grade]:
                            try:
                                daily = _daily(provider, o.symbol, 500)
                                frames = {t: resample_ohlcv(daily, t)
                                          for t in ("daily", "weekly", "monthly")}
                                ctx = assemble_context(o.symbol, daily,
                                                       with_confluence_frames=frames)
                                g = desk.grade(o.symbol, ctx, side=o.side)
                                st.markdown(
                                    f"### {_grade_badge(g.grade)} &nbsp; {g.symbol} "
                                    f"<span class='nt-pill'>panel {g.composite:.0f}/100</span>",
                                    unsafe_allow_html=True)
                                st.caption(g.summary)
                                vrows = [{"agent": v.title, "score": v.score,
                                          "stance": v.stance} for v in g.verdicts]
                                st.dataframe(vrows, use_container_width=True,
                                             hide_index=True)
                            except Exception as exc:  # noqa: BLE001
                                st.write(f"{o.symbol}: skipped ({exc})")
                    st.caption("Panel grades reflect evidence + scrutiny, not a "
                               "profit guarantee. Educational only.")


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

            show_fib = st.checkbox("Overlay Fibonacci retracement", value=False,
                                   key="an_fib",
                                   help="auto-drawn on the dominant swing — "
                                        "industry-standard 23.6/38.2/50/61.8/78.6% levels")
            fig, notes = build_figure(f"{symbol} · {timeframe}", df, bars=220,
                                      show_fib=show_fib)
            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom": True, "displaylogo": False})

            _tcfg_v = ThesisConfig.from_config(cfg)
            if _tcfg_v.enabled and st.button("👁 AI chart read (vision)",
                                             key="an_vision",
                                             help="Claude looks at the chart image "
                                                  "and gives an analyst read"):
                with st.spinner(f"Claude ({_tcfg_v.model}) is reading the chart…"):
                    try:
                        from nsetrade.charts import render_chart_bytes
                        png = render_chart_bytes(f"{symbol} ({timeframe})", df, bars=200)
                        st.markdown(ThesisWriter(_tcfg_v).read_chart(png, symbol))
                        st.caption("AI vision read of the chart image. "
                                   "Educational only — not investment advice.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"chart read failed: {exc}")

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
                # Fibonacci levels (retracement + extension targets)
                from nsetrade.fibonacci import fib_extension, fib_retracement
                fr = fib_retracement(df)
                if fr.found:
                    st.markdown("**Fibonacci retracement** "
                                f"({fr.direction}-swing)")
                    role = "support" if fr.direction == "up" else "resistance"
                    frows = [{"level": l.label, "price": round(l.price, 2),
                              "←": "nearest" if l is fr.nearest else ""}
                             for l in fr.levels]
                    st.dataframe(frows, use_container_width=True, hide_index=True)
                    st.caption(f"These act as {role}. Nearest: "
                               f"{fr.nearest.label} @ {fr.nearest.price:.1f}")
                    ext = fib_extension(df)
                    if ext.found:
                        tg = ", ".join(f"{l.label} @ {l.price:.1f}"
                                       for l in ext.levels if l.ratio >= 1.0)
                        st.caption(f"Extension targets (upside): {tg}")
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
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.warning("No patterns detected.")


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
