"""Streamlit dashboard for nsetrade.

Run with:

    pip install -e ".[dashboard]"
    streamlit run dashboard/app.py

Three tabs:
  • Analyse   — chart + indicators + scored signal for one stock
  • Screener  — rank a universe by signal score
  • Backtest  — risk-managed backtest with an equity curve

The provider (yfinance / kite) and credentials come from config.yaml, exactly
like the CLI.
"""

from __future__ import annotations

import streamlit as st

from nsetrade.backtest import backtest_risk
from nsetrade.config import load_config, provider_config
from nsetrade.data import get_provider
from nsetrade.indicators import add_all
from nsetrade.patterns.candlestick import detect_candlesticks
from nsetrade.patterns.chart import detect_chart_patterns
from nsetrade.screener import screen
from nsetrade.signals.engine import signal_for_frame
from nsetrade.universe import UNIVERSES, get_universe

st.set_page_config(page_title="nsetrade — NSE analysis", layout="wide")
cfg = load_config()


@st.cache_data(show_spinner=False)
def _history(provider, symbol, days):
    prov = get_provider(provider, provider_config(cfg, provider))
    return prov.history(symbol, period_days=days)


def _get(sig, key, default=float("nan")):
    val = sig.indicators.get(key)
    return val if val is not None else default


# ---- sidebar -------------------------------------------------------------
st.sidebar.title("📈 nsetrade")
provider = st.sidebar.selectbox(
    "Data provider", ["yfinance", "kite"],
    index=0 if cfg.get("default_provider") != "kite" else 1,
)
st.sidebar.caption(
    "Credentials for `kite` are read from config.yaml. "
    "yfinance needs no setup."
)
st.sidebar.warning("Research/education only — not investment advice.")

tab_analyse, tab_screen, tab_backtest = st.tabs(
    ["Analyse", "Screener", "Backtest"]
)


# ---- Analyse tab ---------------------------------------------------------
with tab_analyse:
    st.subheader("Analyse a stock")
    col1, col2 = st.columns([2, 1])
    symbol = col1.text_input("NSE symbol", value="RELIANCE").strip().upper()
    days = col2.slider("History (days)", 120, 800, 400, step=20)

    if st.button("Analyse", type="primary") or symbol:
        try:
            df = _history(provider, symbol, days)
            sig = signal_for_frame(symbol, df)
            enriched = add_all(df)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Close", f"{sig.close:,.2f}")
            c2.metric("Signal", sig.verdict, f"{sig.score:+.2f}")
            c3.metric("RSI(14)", f"{_get(sig, 'rsi_14'):.1f}")
            c4.metric("ADX(14)", f"{_get(sig, 'adx'):.1f}")

            view = enriched.tail(min(len(enriched), 250))
            st.line_chart(
                view[["close", "sma_20", "sma_50", "sma_200"]],
                height=320,
            )
            cc1, cc2 = st.columns(2)
            cc1.line_chart(view[["rsi_14"]], height=160)
            cc2.line_chart(view[["macd", "macd_signal"]], height=160)

            st.markdown("**Why this signal:**")
            st.write(", ".join(sig.reasons) if sig.reasons else "no notable patterns")
            st.markdown(
                f"**Support:** {sig.levels.get('support')}  \n"
                f"**Resistance:** {sig.levels.get('resistance')}"
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not analyse {symbol}: {exc}")


# ---- Screener tab --------------------------------------------------------
with tab_screen:
    st.subheader("Screen a universe")
    uni = st.selectbox("Universe", list(UNIVERSES), index=0)
    top = st.slider("Show top N per side", 5, 30, 10)
    if st.button("Run screen"):
        symbols = get_universe(uni)
        prog = st.progress(0.0, text="scanning…")

        def _cb(done, total, sym):
            prog.progress(done / total, text=f"scanning {sym} ({done}/{total})")

        result = screen(
            symbols, provider=provider,
            provider_config=provider_config(cfg, provider),
            on_progress=_cb,
        )
        prog.empty()
        df = result.to_frame()
        if df.empty:
            st.warning("No results (data errors). Try the yfinance provider.")
        else:
            st.markdown("**Top bullish**")
            st.dataframe(df.head(top), use_container_width=True)
            st.markdown("**Top bearish**")
            st.dataframe(df.tail(top).iloc[::-1], use_container_width=True)
        if result.errors:
            st.caption(f"{len(result.errors)} symbols skipped due to errors.")


# ---- Backtest tab --------------------------------------------------------
with tab_backtest:
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
            df = _history(provider, bsym, int(years * 365) + 30)
            res = backtest_risk(
                bsym, df, strategy=strat, stop_atr=stop_atr,
                target_atr=target_atr or None, risk_per_trade=risk,
            )
            m = res.metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total return", f"{m['total_return']:+.1%}")
            m2.metric("CAGR", f"{m['cagr']:+.1%}")
            m3.metric("Sharpe", f"{m['sharpe']:.2f}")
            m4.metric("Max drawdown", f"{m['max_drawdown']:.1%}")
            st.line_chart(res.equity_curve, height=300)
            st.code(res.summary())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest failed: {exc}")
