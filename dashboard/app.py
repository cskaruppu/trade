"""EdgeForge — professional local trading-analysis dashboard.

Run with:
    pip install -e ".[dashboard]"
    streamlit run dashboard/app.py

Grouped sidebar navigation:
  Discover — Home, Opportunities, Ask AI (natural-language screener)
  Analyse  — Analyse (chart + patterns + Fibonacci + AI read), Confluence,
             Pattern Edge, Screener
  Manage   — Watchlist, Backtest

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

try:  # added later — guarded so a stale universe.py degrades, not crashes
    from nsetrade.universe import refresh_nse_equity_list
except ImportError:
    refresh_nse_equity_list = None

st.set_page_config(page_title="EdgeForge", page_icon="⚡", layout="wide")
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


def _pattern_confidence(row: dict):
    """Derive a confidence label + colour from a Pattern-Picks result row.

    Based on the *evidence*: out-of-sample robustness, historical win-rate,
    sample size, and volume confirmation. Not a prediction — a trust rating.
    """
    import re

    robust = row.get("robust") == "✓"
    vol = row.get("vol") == "✓"
    m = re.match(r"(\d+)%\s*/\s*(\d+)", str(row.get("edge", "")))
    win = int(m.group(1)) if m else 0
    occ = int(m.group(2)) if m else 0
    if robust and win >= 60 and occ >= 5:
        return ("High confidence" + (" + volume" if vol else ""), "#26a69a")
    if robust and occ >= 3:
        return ("Moderate confidence", "#e3b341")
    if occ and occ < 3:
        return ("Low confidence — tiny sample", "#ef5350")
    return ("Unproven edge", "#ef5350")


@st.cache_data(show_spinner=False)
def _best_pattern_badge(symbol, days, timeframe):
    """Find the strongest detected pattern for a stock + its validated edge,
    and return (badge_text, colour) for annotating its chart. Cached per stock."""
    try:
        from nsetrade.ai import _pattern_key
        from nsetrade.edge import pattern_edge_validated
        from nsetrade.patterns.advanced import detect_advanced, volume_confirms
        df = _history(provider, symbol, days, timeframe)
        matches = detect_advanced(df)
        if not matches:
            return None
        m = next((x for x in matches if x.status == "breakout"), matches[0])
        row = {"pattern": m.name, "edge": "-", "robust": "-",
               "vol": "✓" if (m.status == "breakout" and volume_confirms(df)) else "-"}
        key = _pattern_key(m.name)
        if key:
            ve = pattern_edge_validated(df, key)
            if ve.full.occurrences:
                row["edge"] = f"{ve.full.win_rate:.0%} / {ve.full.occurrences}"
                row["robust"] = "✓" if ve.robust else "✗"
        conf, color = _pattern_confidence(row)
        return (f"{row['pattern']}  ·  edge {row['edge']}  ·  robust {row['robust']}"
                f"  ·  vol {row['vol']}  ·  {conf}"), color
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False)
def _daily(provider, symbol, days):
    prov = get_provider(provider, provider_config(cfg, provider))
    return prov.history(symbol, period_days=days)


def _history(provider, symbol, days, timeframe="daily"):
    df = _daily(provider, symbol, scale_period_days(days, timeframe))
    return resample_ohlcv(df, timeframe)


@st.cache_data(show_spinner=False)
def _symbol_options():
    """Searchable 'SYMBOL — Company' options + a label→symbol map."""
    from nsetrade.universe import symbol_choices
    options, sym_by = [], {}
    for s, n in symbol_choices():
        label = f"{s} — {n}" if n else s
        options.append(label)
        sym_by[label] = s
    return options, sym_by


@st.cache_data(show_spinner=False, ttl=3600)
def _fundamentals(symbol):
    from nsetrade.fundamentals import fetch_fundamentals
    return fetch_fundamentals(symbol)


@st.cache_data(show_spinner=False, ttl=3600)
def _news(symbol, name=None):
    from nsetrade.fundamentals import fetch_all_news
    return fetch_all_news(symbol, name=name, limit=8)


def _debt_color(status: str) -> str:
    if "Debt-free" in status or "Near" in status:
        return "#26a69a"
    if "Low" in status or "Moderate" in status:
        return "#e3b341"
    if "High" in status:
        return "#ef5350"
    return "#90a4ae"


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
st.sidebar.title("⚡ EdgeForge")
st.sidebar.caption("Forge an edge from evidence — runs only on this machine.")
_PROVIDERS = ["yfinance", "bhavcopy"]
provider = st.sidebar.selectbox(
    "Data provider", _PROVIDERS,
    index=_PROVIDERS.index(cfg.get("default_provider", "yfinance"))
    if cfg.get("default_provider", "yfinance") in _PROVIDERS else 0,
)
timeframe = st.sidebar.radio("Timeframe", ["daily", "weekly", "monthly"],
                             horizontal=True)
st.sidebar.caption("yfinance is free (no key). bhavcopy reads your local "
                   "Bhavcopy store (run `nsetrade fetch-bhavcopy`).")

with st.sidebar.expander("➕ Full NSE coverage (~2000 stocks)"):
    st.caption("Index lists cover up to Nifty 500. Download the full NSE equity "
               "list once to scan **everything** as the 'nse_all' universe — it "
               "then appears in every dropdown.")
    if refresh_nse_equity_list is None:
        st.warning("Your local nsetrade is out of date. Update it with "
                   "`scripts\\update.bat` (or `git pull`) to enable this.")
    elif st.button("Download / refresh full NSE list"):
        with st.spinner("Downloading the NSE equity list…"):
            try:
                syms = refresh_nse_equity_list()
                st.success(f"Added {len(syms)} NSE stocks as 'nse_all'. "
                           "Pick it in any Universe dropdown.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Download failed: {exc}")
    st.caption("Tip: scanning ~2000 stocks live is slow — run "
               "`nsetrade precompute --universe nse_all --with-patterns` nightly "
               "and use the cached/auto-loaded results.")
    st.caption("For reliable full-NSE data without per-stock API limits, build a "
               "local store: `nsetrade fetch-bhavcopy --days 400`, then pick the "
               "**bhavcopy** data provider above.")

st.sidebar.warning("Research/education only — not investment advice.")

# ---- hero header + KPI strip ----------------------------------------------
st.markdown(
    """
    <div class="nt-hero">
      <h1>⚡ EdgeForge</h1>
      <div class="tag">Forge an edge from evidence · AI analyst desk · runs 100% on your machine</div>
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

# ---- grouped sidebar navigation (industry-standard left nav) ---------------
_NAV_GROUPS = {
    "Discover": ["🏠 Home", "🚀 Opportunities", "🏆 Pattern Picks",
                 "📈 Breakouts", "💬 Ask AI"],
    "Analyse": ["🔬 Deep Dive", "📊 Analyse", "🎯 Confluence",
                "🧪 Pattern Edge", "🔎 Screener"],
    "Manage": ["📋 Track Record", "⭐ Watchlist", "📉 Backtest"],
}
_PAGES = [p for group in _NAV_GROUPS.values() for p in group]
if st.session_state.get("nav_page") not in _PAGES:
    st.session_state["nav_page"] = _PAGES[0]
st.sidebar.divider()

# bootstrap icon per page (used by the polished option_menu when available)
_ICONS = {
    "🏠 Home": "house", "🚀 Opportunities": "rocket-takeoff",
    "🏆 Pattern Picks": "trophy", "📈 Breakouts": "graph-up-arrow",
    "💬 Ask AI": "chat-dots", "🔬 Deep Dive": "search-heart",
    "📊 Analyse": "bar-chart-line",
    "🎯 Confluence": "bullseye", "🧪 Pattern Edge": "clipboard-data",
    "🔎 Screener": "search", "📋 Track Record": "clipboard-check",
    "⭐ Watchlist": "star", "📉 Backtest": "graph-down",
}

try:
    from streamlit_option_menu import option_menu
    _HAS_OPTION_MENU = True
except ImportError:
    _HAS_OPTION_MENU = False

if _HAS_OPTION_MENU:
    _plain = [p.split(" ", 1)[1] for p in _PAGES]          # strip the emoji
    with st.sidebar:
        _sel = option_menu(
            "Menu", _plain,
            icons=[_ICONS.get(p, "dot") for p in _PAGES],
            menu_icon="lightning-charge-fill",
            default_index=_PAGES.index(st.session_state["nav_page"]),
            styles={
                "container": {"background-color": "#11151c", "padding": "4px"},
                "icon": {"color": "#79c7bd", "font-size": "0.95rem"},
                "nav-link": {"font-size": "0.92rem", "color": "#cfd8dc",
                             "--hover-color": "#1b2129"},
                "nav-link-selected": {"background-color": "#26a69a",
                                      "color": "#06120f", "font-weight": "600"},
            })
    _page = next(p for p in _PAGES if p.split(" ", 1)[1] == _sel)
    st.session_state["nav_page"] = _page
else:
    # fallback: grouped section headers + full-width buttons (no extra dep)
    st.sidebar.markdown("### Menu")
    for _grp, _items in _NAV_GROUPS.items():
        st.sidebar.caption(_grp.upper())
        for _it in _items:
            _active = st.session_state["nav_page"] == _it
            if st.sidebar.button(_it, key=f"nav_{_it}", use_container_width=True,
                                 type="primary" if _active else "secondary"):
                st.session_state["nav_page"] = _it
                st.rerun()
    _page = st.session_state["nav_page"]


# ---- Home -----------------------------------------------------------------
if _page == "🏠 Home":
    st.subheader("Today at a glance")
    hit = None
    try:
        from nsetrade.scan_cache import ScanCache
        for kind, uni in [("opportunities", "nifty100:long"),
                          ("opportunities", "nifty50:long")]:
            hit = ScanCache().latest(kind, uni)
            if hit:
                break
    except Exception:  # noqa: BLE001
        hit = None
    if hit and hit["rows"]:
        from datetime import datetime
        when = datetime.fromtimestamp(hit["created_at"]).strftime("%d %b %H:%M")
        st.markdown(f"**Top setups from your last scan** ({when})")
        st.dataframe(hit["rows"][:10], use_container_width=True, hide_index=True)
        st.caption("Open **🚀 Opportunities** to scan live, or **💬 Ask AI** to "
                   "search in plain English.")
    else:
        st.info("No cached scan yet. Get started:")
        st.markdown(
            "- **🚀 Opportunities** — rank the most tradeable setups in a universe\n"
            "- **💬 Ask AI** — describe what you want; Claude builds the filter\n"
            "- **📊 Analyse** — deep-dive one stock (chart, patterns, Fibonacci, AI read)\n"
            "- Run `nsetrade precompute` nightly so this page loads instant rankings")
    st.divider()
    st.caption("EdgeForge · evidence-based, AI-native, private. "
               "Educational only — not investment advice.")


# ---- Deep Dive (single-stock full report) ----------------------------------
if _page == "🔬 Deep Dive":
    st.subheader("🔬 Single-stock deep dive")
    st.caption("One stock, full report: which pattern it's forming (drawn on the "
               "chart), the confirmed entry / stop / target, its historical edge, "
               "and an AI read — over the full chart history.")
    dd1, dd2, dd3 = st.columns([2, 1, 1])
    _opts, _sym_by = _symbol_options()
    _default = next((i for i, o in enumerate(_opts)
                     if o.startswith("RELIANCE")), 0)
    _pick = dd1.selectbox("Search NSE stock (type a symbol or company name)",
                          _opts, index=_default, key="dd_pick")
    dsym = _sym_by.get(_pick, _pick).strip().upper()
    if len(_opts) < 200:
        dd1.caption("💡 Showing index stocks only — download the full NSE list "
                    "(sidebar → Full NSE coverage) to search all ~2000.")
    dtf = dd2.radio("Timeframe", ["daily", "weekly", "monthly"],
                    horizontal=True, key="dd_tf")
    dcap = dd3.number_input("Capital (₹)", value=100000, step=10000, key="dd_cap")

    if dsym and st.button("Analyse stock", type="primary", key="dd_go"):
        st.session_state["dd_run"] = {"sym": dsym, "tf": dtf, "cap": float(dcap)}

    run = st.session_state.get("dd_run")
    if run:
        from nsetrade.ai import _pattern_key
        from nsetrade.edge import pattern_edge_validated
        from nsetrade.patterns import detect_advanced
        from nsetrade.tradeplan import trade_plan
        sym, tf, cap = run["sym"], run["tf"], run["cap"]
        try:
            # fetch generous history so "Max" can show the stock's whole life
            tf_days = {"daily": 4000, "weekly": 9000, "monthly": 14000}[tf]
            daily = _daily(provider, sym, tf_days)
            df = resample_ohlcv(daily, tf)
            sig = signal_for_frame(sym, df)
            matches = detect_advanced(df)
            # pick the primary pattern: a bullish breakout if present, else strongest
            best = None
            for m in matches:
                if m.direction == "bullish" and m.status == "breakout":
                    best = m
                    break
            if best is None and matches:
                best = matches[0]
            conf = edge_row = ve = None      # set below when a pattern is found

            # headline: what pattern + verdict
            if best:
                key = _pattern_key(best.name)
                # bound the walk-forward to a solid sample (keeps it fast even
                # when the chart shows full history)
                _edge_bars = {"daily": 1500, "weekly": 600, "monthly": 300}[tf]
                ve = (pattern_edge_validated(df.tail(_edge_bars), key)
                      if key else None)
                edge_row = {
                    "pattern": best.name,
                    "edge": (f"{ve.full.win_rate:.0%} / {ve.full.occurrences}"
                             if ve and ve.full.occurrences else "-"),
                    "robust": ("✓" if ve and ve.robust else
                               "✗" if ve else "-"),
                    "vol": "✓" if best.volume_confirmed else "-",
                }
                conf, color = _pattern_confidence(edge_row)
                st.markdown(
                    f"### {sym} — **{best.name}** "
                    f"<span class='nt-pill' style='border-color:{color};color:{color}'>"
                    f"{conf}</span>", unsafe_allow_html=True)
                st.caption(f"Status: **{best.status}** · direction: {best.direction} "
                           f"· edge {edge_row['edge']} · robust {edge_row['robust']} "
                           f"· volume {edge_row['vol']} · signal: {sig.verdict}")
            else:
                st.markdown(f"### {sym} — no clear structural pattern")
                st.caption(f"Signal: {sig.verdict} (score {sig.score:+.2f}). "
                           "Showing trend + levels instead.")

            # ---- accuracy & historical edge of the pattern ----
            if best and ve and ve.full.occurrences:
                st.markdown("#### Accuracy & historical edge")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Accuracy (historical)", f"{ve.full.win_rate:.0%}",
                          help="how often this pattern was followed by a positive "
                               "move on THIS stock")
                oos = (f"{ve.out_sample.win_rate:.0%}" if ve.out_sample.occurrences
                       else "—")
                a2.metric("Out-of-sample", oos,
                          help="accuracy on data the test never saw — the honest "
                               "number; ✓ robust means it held up")
                a3.metric("Occurrences", ve.full.occurrences)
                a4.metric("Avg forward return", f"{ve.full.avg_return:+.1%}")
                st.caption(f"Verdict: **{ve.verdict}**. Accuracy = historical "
                           "hit-rate, not a prediction — small samples are noisy "
                           "and markets change.")
            elif best:
                st.caption("No measurable historical sample for this pattern on "
                           "this stock yet — treat the setup as unproven.")

            # ---- history depth control + chart (only the primary pattern drawn) ----
            _per_year = {"daily": 252, "weekly": 52, "monthly": 12}[tf]
            hsel = st.select_slider(
                "Chart history", options=["1Y", "3Y", "5Y", "10Y", "Max"],
                value="3Y", key="dd_hist")
            _bars = (len(df) if hsel == "Max"
                     else {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}[hsel] * _per_year)
            _bars = max(60, min(_bars, len(df)))
            only = {_pattern_key(best.name)} if best else None
            fig, notes = build_figure(f"{sym} · {tf}", df, bars=_bars,
                                      show_patterns=True, show_fib=False,
                                      only_keys=only)

            # trade plan → entry / stop / target, drawn on the chart
            direction = "long" if (best.direction == "bullish" if best
                                   else sig.score >= 0) else "short"
            plan = None
            try:
                plan = trade_plan(sym, df, direction=direction, capital=cap,
                                  risk_pct=0.01, pattern=best)
                fig.add_hline(y=plan.entry, line=dict(color="#42a5f5", width=1.2,
                              dash="dash"), row=1, col=1,
                              annotation_text=f"Entry {plan.entry:.1f}",
                              annotation_position="bottom left",
                              annotation_font_color="#42a5f5")
                fig.add_hline(y=plan.stop, line=dict(color="#ef5350", width=1.2,
                              dash="dash"), row=1, col=1,
                              annotation_text=f"Stop {plan.stop:.1f}",
                              annotation_position="bottom left",
                              annotation_font_color="#ef5350")
            except Exception:  # noqa: BLE001
                pass

            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom": True, "displaylogo": False})

            # explicit trade levels
            if plan:
                st.markdown("#### Trade plan")
                t1, t2, t3, t4, t5 = st.columns(5)
                t1.metric("Entry", f"₹{plan.entry:,.2f}")
                t2.metric("Stop loss", f"₹{plan.stop:,.2f}")
                t3.metric("Target", f"₹{plan.target:,.2f}")
                t4.metric("Reward:Risk", f"{plan.rr:.1f} : 1")
                t5.metric("Size (1% risk)", f"{plan.shares:,}")
                st.caption("Entry = pattern breakout · Stop = below structure/ATR · "
                           "Target = measured move. Size risks 1% of capital. "
                           "Educational — confirm before trading.")
            if notes:
                tnote = [n for n in notes if "Upside potential" in n]
                if tnote:
                    st.success("🎯 " + tnote[0])

            # ---- fundamentals & news (best-effort, via Yahoo) ----
            st.divider()
            st.markdown("#### Fundamentals & news")
            fund = _fundamentals(sym)
            news = _news(sym, fund.name if fund else None)
            if fund:
                dc = _debt_color(fund.debt_status)
                st.markdown(
                    f"<span class='nt-pill' style='border-color:{dc};color:{dc}'>"
                    f"{fund.debt_status}</span>", unsafe_allow_html=True)
                fc = st.columns(4)
                fc[0].metric("Market cap",
                             f"₹{fund.market_cap / 1e7:,.0f} Cr" if fund.market_cap
                             else "—")
                fc[1].metric("P/E", f"{fund.pe:.1f}" if fund.pe is not None else "—")
                fc[2].metric("ROE", f"{fund.roe:.0%}" if fund.roe is not None else "—")
                fc[3].metric("Sector", fund.sector or "—")
                st.caption(" · ".join(fund.highlights))
            else:
                st.caption("Fundamentals unavailable for this symbol (Yahoo returned "
                           "nothing — common for some NSE stocks).")
            sent = st.session_state.get("dd_sent", {}).get(sym)
            if news:
                st.markdown("**Recent headlines** (Google News + Yahoo)")
                _dots = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
                for i, nws in enumerate(news):
                    dot = _dots.get(sent[i] if sent and i < len(sent) else None, "")
                    pub = nws.get("publisher") or ""
                    src = nws.get("source") or ""
                    tag = f"_{pub}_ · {src}" if pub else src
                    title = (f"[{nws['title']}]({nws['link']})" if nws.get("link")
                             else nws["title"])
                    st.markdown(f"- {dot} {title} — {tag}")
                _tcn = ThesisConfig.from_config(cfg)
                if sent:
                    pos, neg = sent.count("positive"), sent.count("negative")
                    mood = ("🟢 net positive" if pos > neg else
                            "🔴 net negative" if neg > pos else "⚪ mixed")
                    st.caption(f"News mood: **{mood}** "
                               f"({pos} positive / {neg} negative / "
                               f"{sent.count('neutral')} neutral)")
                elif _tcn.enabled and st.button("🧠 Tag news sentiment (AI)",
                                                key="dd_sent_btn"):
                    with st.spinner("Claude is reading the headlines…"):
                        try:
                            s = ThesisWriter(_tcn).classify_headlines(
                                [n["title"] for n in news])
                            st.session_state.setdefault("dd_sent", {})[sym] = s
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"sentiment tagging failed: {exc}")
            else:
                st.caption("No recent news found for this symbol.")
            st.caption("ℹ️ News aggregated from Google News (many publishers) + "
                       "Yahoo. Fundamentals from Yahoo — free but **unofficial** "
                       "and may be incomplete or stale. Verify before acting.")

            # AI analysis
            st.divider()
            _tcfg_dd = ThesisConfig.from_config(cfg)
            if not _tcfg_dd.enabled:
                st.caption("💡 Add an Anthropic API key for a full AI read "
                           "(text thesis + a vision read of this chart).")
            else:
                dac1, dac2 = st.columns(2)
                if dac1.button("🤖 AI analysis & thesis", key="dd_ai"):
                    with st.spinner(f"Claude ({_tcfg_dd.model}) is analysing {sym}…"):
                        try:
                            frames = {t: resample_ohlcv(daily, t)
                                      for t in ("daily", "weekly", "monthly")}
                            ctx = assemble_context(sym, daily,
                                                   with_confluence_frames=frames)
                            if fund:
                                ctx["fundamentals"] = fund.summary()
                            if news:
                                ctx["news"] = [n["title"] for n in news]
                            _thesis = ThesisWriter(_tcfg_dd).write(sym, ctx)
                            st.session_state.setdefault("dd_thesis", {})[sym] = _thesis
                            st.markdown(_thesis)
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"AI analysis failed: {exc}")
                if dac2.button("👁 AI chart read (vision)", key="dd_vis"):
                    with st.spinner(f"Claude is reading {sym}'s chart…"):
                        try:
                            from nsetrade.charts import render_chart_bytes
                            png = render_chart_bytes(f"{sym} ({tf})", df, bars=220)
                            st.markdown(ThesisWriter(_tcfg_dd).read_chart(png, sym))
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"chart read failed: {exc}")
            st.caption("AI is educational analysis, not investment advice.")

            # ---- export the report ----
            st.divider()
            from nsetrade.report import build_report_md
            _news_rep = [{"title": n.get("title"), "publisher": n.get("publisher"),
                          "sentiment": sent[i] if sent and i < len(sent) else None}
                         for i, n in enumerate(news or [])]
            md = build_report_md(
                sym, tf,
                pattern=best.name if best else None,
                confidence=conf, edge=edge_row["edge"] if edge_row else None,
                entry=f"{plan.entry:.2f}" if plan else None,
                stop=f"{plan.stop:.2f}" if plan else None,
                target=f"{plan.target:.2f}" if plan else None,
                rr=f"{plan.rr:.1f}:1" if plan else None,
                size=plan.shares if plan else None,
                debt_status=fund.debt_status if fund else None,
                highlights=fund.highlights if fund else None,
                news=_news_rep,
                thesis=st.session_state.get("dd_thesis", {}).get(sym))
            ec1, ec2 = st.columns(2)
            ec1.download_button("⬇ Download report (Markdown)", md,
                                file_name=f"{sym}_edgeforge_report.md",
                                mime="text/markdown")
            try:
                from nsetrade.charts import render_chart_bytes
                _png = render_chart_bytes(f"{sym} ({tf})", df, bars=220)
                ec2.download_button("⬇ Download chart (PNG)", _png,
                                    file_name=f"{sym}_chart.png", mime="image/png")
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not analyse {sym}: {exc}")


# ---- Pattern Picks (focused pattern screener) ------------------------------
if _page == "🏆 Pattern Picks":
    st.subheader("Pattern picks")
    st.caption("Find stocks forming specific chart patterns, each scored by its "
               "**validated historical edge on that stock** (✓ = held up "
               "out-of-sample). The strongest setups rise to the top.")
    _PAT_CHOICES = {
        "Cup & Handle": "cup_and_handle",
        "Darvas Box": "darvas_box",
        "VCP (Volatility Contraction)": "vcp",
        "Accumulation Base (trendline + support)": "accumulation",
        "Bull Flag": "flag",
        "Double Bottom": "double_bottom",
        "Ascending/Descending Triangle": "triangle",
        "Head & Shoulders": "head_shoulders",
    }
    pp1, pp2 = st.columns([2, 1])
    chosen = pp1.multiselect("Patterns", list(_PAT_CHOICES),
                             default=["Cup & Handle", "Darvas Box"], key="pp_pat")
    pp_uni = pp1.selectbox("Universe", list_universes(), index=0, key="pp_uni")
    pp_brk = pp2.checkbox("Confirmed breakouts only", value=False, key="pp_brk")
    pp_vol = pp2.checkbox("Volume-confirmed only", value=False, key="pp_vol",
                          help="breakout backed by above-average volume")
    pp_edge = pp2.checkbox("Score historical edge", value=True, key="pp_edge",
                           help="backtests each pattern on each stock; slower")

    from nsetrade.scan_cache import ScanCache
    _pp_cache = ScanCache()

    if st.button("Scan for patterns", type="primary", disabled=not chosen):
        from nsetrade.pattern_scan import scan_for_patterns
        keys = [_PAT_CHOICES[c] for c in chosen]
        syms = get_universe(pp_uni)
        prog = st.progress(0.0)
        hits, errors = scan_for_patterns(
            syms, keys, provider=provider,
            provider_config=provider_config(cfg, provider),
            with_edge=pp_edge, only_breakouts=pp_brk,
            only_volume_confirmed=pp_vol,
            on_progress=lambda d, t, s: prog.progress(d / t, text=s))
        prog.empty()
        rows = [h.as_row() for h in hits]
        st.session_state["pp_hits"] = rows
        st.session_state["pp_syms"] = [h.symbol for h in hits]
        st.session_state["pp_when"] = None
        st.session_state["pp_loaded_uni"] = pp_uni      # don't let auto-load override
        # cache so the page auto-loads these next time it's opened
        _pp_cache.save_run("patterns", pp_uni, rows, meta={"patterns": chosen})
        _pp_cache.prune(keep_per_key=5)

    # auto-load the cached scan for the SELECTED universe, and re-load whenever
    # the universe changes (so picking nse_all instantly shows its cached picks)
    if st.session_state.get("pp_loaded_uni") != pp_uni:
        from datetime import datetime
        hit = _pp_cache.latest("patterns", pp_uni)
        if hit and hit["rows"]:
            st.session_state["pp_hits"] = hit["rows"]
            st.session_state["pp_syms"] = [r["symbol"] for r in hit["rows"]]
            st.session_state["pp_when"] = datetime.fromtimestamp(
                hit["created_at"]).strftime("%d %b %H:%M")
        else:                                            # no cache for this universe
            for _k in ("pp_hits", "pp_syms", "pp_when"):
                st.session_state.pop(_k, None)
        st.session_state["pp_loaded_uni"] = pp_uni

    hits_rows = st.session_state.get("pp_hits")
    _pp_when = st.session_state.get("pp_when")
    if _pp_when:
        st.info(f"Showing the cached scan for **{pp_uni}** from **{_pp_when}** "
                f"({len(hits_rows)} picks). Click *Scan for patterns* to refresh.")
    if hits_rows is not None:
        if not hits_rows:
            st.warning("No matching patterns found on this universe.")
        else:
            from nsetrade.ai import _pattern_key

            # only show the patterns currently selected in the multiselect above
            _sel_keys = {_PAT_CHOICES[c] for c in chosen}
            rows = [r for r in hits_rows
                    if _pattern_key(r.get("pattern", "")) in _sel_keys]

            # ---- one-click high-conviction big-move filter ----
            hc = st.checkbox(
                "🎯 High-conviction big-move setups only", value=False, key="pp_hc",
                help="confirmed breakout + volume + a robust historical edge "
                     "(High confidence) + an implied move of at least the % below")
            fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
            _q = fc1.text_input("🔍 Search symbol", key="pp_q").strip().upper()
            _conf_pick = fc2.multiselect("Confidence",
                                         ["High", "Moderate", "Low/Unproven"],
                                         default=(["High"] if hc else []),
                                         key="pp_conf")
            _status_pick = fc3.multiselect("Status", ["breakout", "forming"],
                                           default=(["breakout"] if hc else []),
                                           key="pp_status")
            _min_up = fc4.slider("Min implied upside %", 0, 100,
                                 30 if hc else 0, 5, key="pp_minup")

            def _tier(r):
                lab = _pattern_confidence(r)[0]
                return ("High" if lab.startswith("High")
                        else "Moderate" if lab.startswith("Moderate")
                        else "Low/Unproven")

            if _q:
                rows = [r for r in rows if _q in r.get("symbol", "")]
            if _conf_pick:
                rows = [r for r in rows if _tier(r) in _conf_pick]
            if _status_pick:
                rows = [r for r in rows if r.get("status") in _status_pick]
            if _min_up > 0:
                rows = [r for r in rows
                        if r.get("upside %") is not None
                        and r["upside %"] >= _min_up]
            if hc:                       # require volume confirmation too
                rows = [r for r in rows if r.get("vol") == "✓"]

            # sort the filtered view: biggest implied move first within the filter
            rows = sorted(rows, key=lambda r: (r.get("upside %") or 0), reverse=True)

            st.caption(f"Showing **{len(rows)}** of {len(hits_rows)} rows · "
                       f"patterns: {', '.join(chosen) or 'none selected'}"
                       + (" · 🎯 high-conviction big-move filter ON" if hc else ""))

            if not rows:
                st.info("No rows match. Loosen the filters — or if you selected a "
                        "pattern that isn't in the cached scan, click "
                        "*Scan for patterns* to run it live.")
            else:
                import pandas as _pd
                _df = _pd.DataFrame(rows)
                _confs = [_pattern_confidence(r) for r in rows]
                _df.insert(len(_df.columns), "confidence", [c[0] for c in _confs])
                _row_colors = [c[1] for c in _confs]

                def _tint(row):
                    return [f"background-color: {_row_colors[row.name]}22"] * len(row)

                st.dataframe(_df.style.apply(_tint, axis=1),
                             use_container_width=True, hide_index=True)
                st.caption("🟢 high · 🟡 moderate · 🔴 low/unproven — by out-of-sample "
                           "robustness, win-rate, sample size & volume. "
                           "edge = win-rate / occurrences. Probability, not a guarantee.")

            # ---- view a pick's chart with the pattern drawn on it ----
            psel = None
            pp_tf = "daily"
            if rows:
                st.divider()
                st.markdown("### 📈 View the chart with the pattern marked")
                cc1, cc2 = st.columns([2, 1])
                psel = cc1.selectbox("Pick a stock from the filtered results",
                                     [r["symbol"] for r in rows], key="pp_sel")
                pp_tf = cc2.radio("Chart timeframe", ["daily", "weekly", "monthly"],
                                  horizontal=True, key="pp_tf")
            pdaily = None
            if psel:
                try:
                    _tf_days = {"daily": 600, "weekly": 1500, "monthly": 3650}[pp_tf]
                    pdaily = _daily(provider, psel, _tf_days)
                    pchart = resample_ohlcv(pdaily, pp_tf)
                    # only draw the pattern(s) the user selected in the filter
                    _only = {_PAT_CHOICES[c] for c in chosen} or None
                    pfig, pnotes = build_figure(f"{psel} · {pp_tf}", pchart, bars=220,
                                                show_patterns=True, show_fib=False,
                                                only_keys=_only)
                    # confidence badge drawn on the chart from the result row
                    prows = [r for r in rows if r.get("symbol") == psel]
                    if prows:
                        r0 = prows[0]
                        conf, color = _pattern_confidence(r0)
                        badge = (f"{r0['pattern']}  ·  edge {r0['edge']}  ·  "
                                 f"robust {r0['robust']}  ·  vol {r0['vol']}  ·  "
                                 f"{conf}")
                        pfig.add_annotation(
                            xref="paper", yref="paper", x=0.005, y=1.07,
                            xanchor="left", yanchor="top", showarrow=False,
                            text=badge, font=dict(size=12, color=color),
                            bgcolor="rgba(17,21,28,0.92)", bordercolor=color,
                            borderwidth=1, borderpad=6)
                    st.plotly_chart(pfig, use_container_width=True,
                                    config={"scrollZoom": True, "displaylogo": False})
                    if prows:
                        st.markdown(
                            f"<span class='nt-pill' style='border-color:{color}'>"
                            f"{conf}</span> &nbsp; edge = historical win-rate / "
                            f"occurrences · robust ✓ = held out-of-sample · "
                            f"vol ✓ = breakout on above-average volume.",
                            unsafe_allow_html=True)
                    # target details (measured move) surfaced as text
                    _tgt = [n for n in pnotes if "Upside potential" in n]
                    if _tgt:
                        st.success("🎯 " + _tgt[0]
                                   + "  —  measured move = breakout + the pattern's "
                                   "own height. A projection, not a prediction.")
                    if pnotes:
                        st.caption(f"On the **{pp_tf}** chart: pattern shape, breakout "
                                   "level (dashed), shaded target zone. Switch the "
                                   "timeframe above to view weekly / monthly.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"could not draw {psel}: {exc}")

            _tcfg_pp = ThesisConfig.from_config(cfg)
            if pdaily is not None and not _tcfg_pp.enabled:
                st.caption("💡 Add an Anthropic API key for an AI read of this "
                           "pattern (text analysis + a vision read of the chart).")
            elif pdaily is not None:
                ac1, ac2 = st.columns(2)
                if ac1.button("🤖 AI pattern analysis & suggestion", key="pp_ai"):
                    with st.spinner(f"Claude ({_tcfg_pp.model}) is analysing {psel}…"):
                        try:
                            frames = {t: resample_ohlcv(pdaily, t)
                                      for t in ("daily", "weekly", "monthly")}
                            ctx = assemble_context(psel, pdaily,
                                                   with_confluence_frames=frames)
                            st.markdown(ThesisWriter(_tcfg_pp).write(psel, ctx))
                            st.caption("Grounded in the pattern + its historical "
                                       "edge. Educational only — not advice.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"analysis failed: {exc}")
                if ac2.button("👁 AI chart read (vision)", key="pp_vision"):
                    with st.spinner(f"Claude is reading {psel}'s chart…"):
                        try:
                            from nsetrade.charts import render_chart_bytes
                            png = render_chart_bytes(f"{psel}", pdaily, bars=200)
                            st.markdown(ThesisWriter(_tcfg_pp).read_chart(png, psel))
                            st.caption("AI vision read of the chart image. "
                                       "Educational only — not advice.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"chart read failed: {exc}")


# ---- Breakouts (new N-period high scanner) ---------------------------------
if _page == "📈 Breakouts":
    from nsetrade.breakout import PERIODS, scan_breakouts
    st.subheader("New-high breakouts")
    st.caption("Stocks breaking out to a new **closing high** over the period you "
               "choose — the classic momentum screen (3-month, 6-month, "
               "52-week, multi-year, or all-time).")
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    b_period = bc1.selectbox("Breakout period", list(PERIODS), index=3,
                             key="bk_period")          # default 52 weeks
    b_uni = bc2.selectbox("Universe", list_universes(), index=0, key="bk_uni")
    b_tol = bc3.slider("Within % of high", 0.0, 5.0, 0.0, 0.5, key="bk_tol",
                       help="0 = must close above the prior high; raise to catch "
                            "stocks right at the edge") / 100.0
    if st.button("Scan breakouts", type="primary"):
        syms = get_universe(b_uni)
        prog = st.progress(0.0)
        res, errors = scan_breakouts(
            syms, period=b_period, tol=b_tol, provider=provider,
            provider_config=provider_config(cfg, provider),
            on_progress=lambda d, t, s: prog.progress(d / t, text=s))
        prog.empty()
        st.session_state["bk_rows"] = [b.as_row() for b in res]
    bk_rows = st.session_state.get("bk_rows")
    if bk_rows is not None:
        if not bk_rows:
            st.warning("No stocks at a new high for this period.")
        else:
            st.success(f"{len(bk_rows)} stocks breaking out to new "
                       f"{b_period} highs.")
            st.dataframe(bk_rows, use_container_width=True, hide_index=True)
            st.caption("'vs high' +ve = above the prior high · "
                       "'bars since high' = how long since the prior peak. "
                       "Confirm trend + volume before acting.")


# ---- Ask AI (natural-language screener) ------------------------------------
if _page == "💬 Ask AI":
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
if _page == "🚀 Opportunities":
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
if _page == "📊 Analyse":
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
            _badge = _best_pattern_badge(symbol, days, timeframe)
            if _badge:
                _btext, _bcolor = _badge
                fig.add_annotation(
                    xref="paper", yref="paper", x=0.005, y=1.07,
                    xanchor="left", yanchor="top", showarrow=False, text=_btext,
                    font=dict(size=12, color=_bcolor),
                    bgcolor="rgba(17,21,28,0.92)", bordercolor=_bcolor,
                    borderwidth=1, borderpad=6)
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


# ---- Track Record ----------------------------------------------------------
if _page == "📋 Track Record":
    from nsetrade.track_record import TrackRecord
    st.subheader("📋 Signal track record")
    st.caption("Honest proof, over time: every logged signal is measured against "
               "what price actually did — wins **and** losses. This is the "
               "scorecard, not a sales pitch.")
    rec = TrackRecord()

    tc1, tc2, tc3 = st.columns([2, 1, 1])
    tr_uni = tc1.selectbox("Universe to log from", list_universes(), index=0,
                           key="tr_uni")
    if tc2.button("➕ Log current breakouts"):
        from nsetrade.pattern_scan import scan_for_patterns
        syms = get_universe(tr_uni)
        prog = st.progress(0.0)
        hits, _ = scan_for_patterns(
            syms, provider=provider, provider_config=provider_config(cfg, provider),
            only_breakouts=True,
            on_progress=lambda d, t, s: prog.progress(d / t, text=s))
        prog.empty()
        logged = 0
        for h in hits:
            if not h.breakout_level:
                continue
            entry = h.breakout_level
            stop = entry * 0.95
            target = entry + (entry - stop)
            conf = ("High" if h.edge_robust and (h.edge_win_rate or 0) >= 0.6
                    else "Moderate" if h.edge_robust else "Low")
            if rec.log_signal(h.symbol, h.pattern, "long", entry, target, stop,
                              confidence=conf):
                logged += 1
        st.success(f"Logged {logged} new breakout signals.")
    if tc3.button("✅ Evaluate matured"):
        def _fetch(sym):
            return _daily(provider, sym, 200)
        prog = st.progress(0.0)
        n = rec.evaluate(fetch=_fetch, forward_bars=20,
                         on_progress=lambda d, t, s: prog.progress(d / max(t, 1),
                                                                   text=s))
        prog.empty()
        st.success(f"Resolved {n} matured signals.")

    sc = rec.scorecard()
    if sc.get("n", 0) == 0:
        st.info("No resolved signals yet. **Log current breakouts**, then come "
                "back after ~20 trading days and **Evaluate matured** — the "
                "scorecard fills in as real outcomes arrive. (Best run on a "
                "schedule alongside your nightly precompute.)")
    else:
        m = st.columns(5)
        m[0].metric("Resolved", sc["n"])
        m[1].metric("Win rate", f"{sc['win_rate']:.0%}")
        m[2].metric("Hit target", f"{sc['hit_target_rate']:.0%}")
        m[3].metric("Hit stop", f"{sc['hit_stop_rate']:.0%}")
        pf = sc["profit_factor"]
        m[4].metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
        st.caption(f"Avg return {sc['avg_return']:+.1%} · "
                   f"avg win {sc['avg_win']:+.1%} · avg loss {sc['avg_loss']:+.1%} "
                   "· every signal counted, wins and losses.")

        # equity curve + drawdown — the most visceral proof
        from nsetrade.track_record import max_drawdown
        curve = rec.equity_curve()
        if len(curve) >= 2:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            xs = list(range(1, len(curve) + 1))
            eq = [round((p["equity"] - 1) * 100, 2) for p in curve]   # % gain
            dd = [round(p["drawdown"] * 100, 2) for p in curve]
            efig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.7, 0.3], vertical_spacing=0.05,
                                 subplot_titles=("Equity (equal-weight, %)",
                                                 "Drawdown (%)"))
            efig.add_trace(go.Scatter(x=xs, y=eq, mode="lines", name="Equity",
                                      line=dict(color="#26a69a", width=2),
                                      fill="tozeroy",
                                      fillcolor="rgba(38,166,154,0.12)"),
                           row=1, col=1)
            efig.add_trace(go.Scatter(x=xs, y=dd, mode="lines", name="Drawdown",
                                      line=dict(color="#ef5350", width=1.5),
                                      fill="tozeroy",
                                      fillcolor="rgba(239,83,80,0.15)"),
                           row=2, col=1)
            efig.update_layout(template="plotly_dark", height=380, showlegend=False,
                               margin=dict(l=10, r=10, t=30, b=10))
            efig.update_xaxes(title_text="trade #", row=2, col=1)
            st.plotly_chart(efig, use_container_width=True)
            st.caption(f"Max drawdown: **{max_drawdown(curve):.0%}** · "
                       "equal-weight, sequential — if you'd taken every signal "
                       "in turn. A visualization, not a guaranteed P&L.")

        by = rec.scorecard(by="pattern")
        prows = [{"pattern": k, "trades": v["n"],
                  "win rate": f"{v['win_rate']:.0%}",
                  "avg return": f"{v['avg_return']:+.1%}",
                  "profit factor": ("∞" if v["profit_factor"] == float("inf")
                                    else f"{v['profit_factor']:.2f}")}
                 for k, v in by.items() if v.get("n")]
        if prows:
            st.markdown("**By pattern**")
            st.dataframe(prows, use_container_width=True, hide_index=True)

    # the actual signal log (proof you can audit)
    allsigs = rec.all_signals()
    if allsigs:
        st.markdown("**Signal log** (audit every call)")
        log_rows = [{"symbol": s.symbol, "pattern": s.pattern,
                     "entry_date": s.entry_date, "entry": round(s.entry, 2),
                     "target": round(s.target, 2), "stop": round(s.stop, 2),
                     "confidence": s.confidence, "status": s.status,
                     "outcome": s.outcome or "—",
                     "return": f"{s.ret:+.1%}" if s.ret is not None else "—"}
                    for s in allsigs]
        st.dataframe(log_rows, use_container_width=True, hide_index=True)
    st.caption("Past performance does not guarantee future results. Educational.")


# ---- Watchlist -------------------------------------------------------------
if _page == "⭐ Watchlist":
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
if _page == "🎯 Confluence":
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
if _page == "🧪 Pattern Edge":
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
if _page == "🔎 Screener":
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
if _page == "📉 Backtest":
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
