"""Command-line interface for nsetrade.

Subcommands:
    analyse    Analyse one stock: indicators, patterns and a scored signal.
    screen     Scan a universe and rank stocks by signal score.
    backtest   Risk-managed backtest of a strategy (single stock or portfolio).
    chart      Render an annotated PNG chart with patterns marked.
    watch      Live watchlist scanner that alerts when signals fire (Kite).
    patterns   List the patterns/strategies the toolkit knows about.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import load_config, provider_config


def _fmt_pct(x):
    return f"{x:.1%}" if x is not None else "n/a"


def _resolve_provider(args, cfg):
    provider = args.provider or cfg.get("default_provider", "yfinance")
    return provider, provider_config(cfg, provider)


# --------------------------------------------------------------------------
# analyse
# --------------------------------------------------------------------------


def cmd_analyse(args, cfg):
    from .signals.engine import analyse

    provider, pconf = _resolve_provider(args, cfg)
    sig = analyse(
        args.symbol,
        provider=provider,
        provider_config=pconf,
        period_days=args.days,
    )
    ind = sig.indicators
    print(f"\n=== {sig.symbol}  ({provider})  as of {sig.date.date()} ===")
    print(f"  Close            : {sig.close:.2f}")
    print(f"  Signal           : {sig.verdict}  (score {sig.score:+.2f})")
    print("  Indicators:")
    print(f"    RSI(14)        : {_num(ind.get('rsi_14'))}")
    print(f"    MACD hist      : {_num(ind.get('macd_hist'), 3)}")
    print(f"    ADX(14)        : {_num(ind.get('adx'))}")
    print(f"    ATR(14)        : {_num(ind.get('atr_14'))}  "
          f"(≈ stop distance)")
    print(f"    SMA 50 / 200   : {_num(ind.get('sma_50'))} / "
          f"{_num(ind.get('sma_200'))}")
    print(f"    Bollinger %B   : {_num(ind.get('bb_pct_b'), 2)}")
    lv = sig.levels
    if lv.get("support") or lv.get("resistance"):
        print("  Levels:")
        print(f"    Support        : {lv.get('support')}")
        print(f"    Resistance     : {lv.get('resistance')}")
    print("  Why:")
    if sig.reasons:
        for r in sig.reasons:
            print(f"    • {r}")
    else:
        print("    • no notable patterns on the latest bar")
    print()


def _num(x, n=1):
    try:
        return f"{float(x):.{n}f}" if x is not None and x == x else "n/a"
    except (TypeError, ValueError):
        return "n/a"


# --------------------------------------------------------------------------
# screen
# --------------------------------------------------------------------------


def cmd_screen(args, cfg):
    from .screener import screen
    from .universe import get_universe

    provider, pconf = _resolve_provider(args, cfg)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        universe = args.universe or cfg.get("default_universe", "nifty50")
        symbols = get_universe(universe)

    def progress(done, total, sym):
        print(f"\r  scanning {done}/{total}  {sym:<14}", end="", file=sys.stderr)

    result = screen(
        symbols,
        provider=provider,
        provider_config=pconf,
        period_days=args.days,
        on_progress=progress,
    )
    print("", file=sys.stderr)

    df = result.to_frame()
    if df.empty:
        print("No results. Errors:")
        for sym, err in list(result.errors.items())[:10]:
            print(f"  {sym}: {err}")
        return

    bullish = df.head(args.top)
    bearish = df.tail(args.top).iloc[::-1]

    print(f"\nTop {args.top} BULLISH setups:")
    _print_table(bullish)
    if not args.bullish_only:
        print(f"\nTop {args.top} BEARISH setups:")
        _print_table(bearish)

    if result.errors:
        print(f"\n({len(result.errors)} symbols skipped due to errors; "
              f"e.g. {next(iter(result.errors))})")


def _print_table(df):
    cols = ["symbol", "close", "score", "verdict", "rsi", "adx", "reasons"]
    cols = [c for c in cols if c in df.columns]
    widths = {"symbol": 12, "close": 9, "score": 7, "verdict": 12,
              "rsi": 6, "adx": 6, "reasons": 40}
    header = "  ".join(f"{c:<{widths.get(c, 10)}}" for c in cols)
    print("  " + header)
    print("  " + "-" * len(header))
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if c == "reasons" and isinstance(val, str) and len(val) > 40:
                val = val[:37] + "..."
            cells.append(f"{str(val):<{widths.get(c, 10)}}")
        print("  " + "  ".join(cells))


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------


def cmd_backtest(args, cfg):
    from .data import get_provider

    provider, pconf = _resolve_provider(args, cfg)
    prov = get_provider(provider, pconf)
    days = int(args.years * 365) + 30
    target_atr = args.target_atr or None  # 0 -> disable take-profit

    # ---- portfolio mode ----
    if args.universe or args.symbols:
        from .backtest import backtest_portfolio
        from .universe import get_universe

        if args.symbols:
            syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            syms = get_universe(args.universe)

        frames = {}
        for i, sym in enumerate(syms):
            print(f"\r  fetching {i + 1}/{len(syms)} {sym:<14}", end="",
                  file=sys.stderr)
            try:
                frames[sym] = prov.history(sym, period_days=days)
            except Exception:  # noqa: BLE001
                pass
        print("", file=sys.stderr)
        result = backtest_portfolio(
            frames, strategy=args.strategy,
            stop_atr=args.stop_atr, target_atr=target_atr,
            risk_per_trade=args.risk, cost_bps=args.cost_bps,
        )
        print("\n" + result.summary() + "\n")
        return

    # ---- single-symbol mode ----
    df = prov.history(args.symbol, period_days=days)
    if args.simple:
        from .backtest import backtest
        result = backtest(args.symbol, df, strategy=args.strategy,
                          cost_bps=args.cost_bps)
        print("\n" + result.summary() + "\n")
        if result.metrics["total_return"] < result.metrics["buy_hold_return"]:
            print("  Note: this strategy underperformed buy-and-hold over this "
                  "window.\n        An edge must beat buy-and-hold *after* costs "
                  "to be worth trading.\n")
    else:
        from .backtest import backtest_risk
        result = backtest_risk(
            args.symbol, df, strategy=args.strategy,
            stop_atr=args.stop_atr, target_atr=target_atr,
            risk_per_trade=args.risk, cost_bps=args.cost_bps,
        )
        print("\n" + result.summary() + "\n")


def cmd_chart(args, cfg):
    from .charts import render_chart
    from .data import get_provider

    provider, pconf = _resolve_provider(args, cfg)
    prov = get_provider(provider, pconf)
    df = prov.history(args.symbol, period_days=args.days)
    out = render_chart(args.symbol, df, out_path=args.out, bars=args.bars)
    print(f"saved chart -> {out}")


def cmd_watch(args, cfg):
    from .live import watch

    provider, pconf = _resolve_provider(args, cfg)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        from .universe import get_universe
        symbols = get_universe(args.universe or cfg.get("default_universe",
                                                        "nifty50"))
    watch(
        symbols,
        provider=provider,
        provider_config=pconf,
        interval_seconds=args.interval,
        only_market_hours=not args.always,
        max_iterations=args.iterations,
    )


# --------------------------------------------------------------------------
# patterns
# --------------------------------------------------------------------------


def cmd_patterns(args, cfg):
    from .patterns.candlestick import CANDLESTICK_PATTERNS
    from .patterns.chart import CHART_PATTERNS
    from .backtest import list_strategies

    def bias(b):
        return {1: "bullish", -1: "bearish", 0: "neutral"}[b]

    print("\nChart patterns / signals:")
    for _, label, b in CHART_PATTERNS:
        print(f"  • {label:<34} ({bias(b)})")
    print("\nCandlestick patterns:")
    for _, label, b in CANDLESTICK_PATTERNS:
        print(f"  • {label:<34} ({bias(b)})")
    print("\nBacktest strategies:")
    for s in list_strategies():
        print(f"  • {s}")
    print()


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nsetrade",
        description="NSE stock pattern, signal, screening & backtesting toolkit.",
    )
    p.add_argument("--version", action="version", version=f"nsetrade {__version__}")
    p.add_argument("--config", help="path to config.yaml (optional)")
    p.add_argument("--provider", help="data provider: yfinance | kite "
                                       "(overrides config)")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyse", help="analyse one stock")
    a.add_argument("symbol", help="NSE symbol, e.g. RELIANCE")
    a.add_argument("--days", type=int, default=400, help="history window (days)")
    a.set_defaults(func=cmd_analyse)

    s = sub.add_parser("screen", help="scan & rank a universe")
    s.add_argument("--universe", help="nifty50 | nifty100 | nifty500")
    s.add_argument("--symbols", help="comma-separated custom symbols")
    s.add_argument("--top", type=int, default=10, help="rows per side")
    s.add_argument("--days", type=int, default=400, help="history window (days)")
    s.add_argument("--bullish-only", action="store_true",
                   help="hide bearish table")
    s.set_defaults(func=cmd_screen)

    b = sub.add_parser("backtest",
                       help="backtest a strategy (risk-managed by default)")
    b.add_argument("symbol", nargs="?", default="RELIANCE",
                   help="NSE symbol (ignored in portfolio mode)")
    b.add_argument("--strategy", default="rsi_ma",
                   help="rsi_ma | sma_crossover | macd | breakout")
    b.add_argument("--years", type=float, default=3.0, help="lookback in years")
    b.add_argument("--cost-bps", type=float, default=5.0,
                   help="per-trade cost in basis points")
    b.add_argument("--stop-atr", type=float, default=2.0,
                   help="stop-loss distance in ATR multiples")
    b.add_argument("--target-atr", type=float, default=4.0,
                   help="take-profit distance in ATR multiples (0 to disable)")
    b.add_argument("--risk", type=float, default=0.01,
                   help="fraction of equity risked per trade (0.01 = 1%%)")
    b.add_argument("--simple", action="store_true",
                   help="use the simple always-in vectorised backtest instead")
    b.add_argument("--universe", help="portfolio mode: nifty50 | nifty100")
    b.add_argument("--symbols", help="portfolio mode: comma-separated symbols")
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("chart", help="render an annotated PNG chart")
    c.add_argument("symbol", help="NSE symbol, e.g. RELIANCE")
    c.add_argument("--out", help="output PNG path (default <SYMBOL>.png)")
    c.add_argument("--days", type=int, default=400, help="history window (days)")
    c.add_argument("--bars", type=int, default=180, help="bars to plot")
    c.set_defaults(func=cmd_chart)

    w = sub.add_parser("watch", help="live watchlist scanner (needs Kite)")
    w.add_argument("--universe", help="nifty50 | nifty100")
    w.add_argument("--symbols", help="comma-separated watchlist")
    w.add_argument("--interval", type=int, default=300,
                   help="seconds between scans")
    w.add_argument("--always", action="store_true",
                   help="scan even outside NSE market hours")
    w.add_argument("--iterations", type=int, default=None,
                   help="stop after N scans (default: run forever)")
    w.set_defaults(func=cmd_watch)

    pt = sub.add_parser("patterns", help="list known patterns & strategies")
    pt.set_defaults(func=cmd_patterns)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(getattr(args, "config", None))
    try:
        args.func(args, cfg)
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
