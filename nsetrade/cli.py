"""Command-line interface for nsetrade.

Subcommands:
    analyse    Analyse one stock: indicators, patterns and a scored signal.
    screen     Scan a universe and rank stocks by signal score.
    backtest   Risk-managed backtest of a strategy (single stock or portfolio).
    chart      Render an annotated PNG chart with patterns marked.
    watch      Polling watchlist scanner that alerts when signals fire.
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


def _fetch(prov, symbol, timeframe, period_days):
    """Fetch daily data and resample to the requested timeframe."""
    from .resample import resample_ohlcv, scale_period_days

    df = prov.history(symbol, period_days=scale_period_days(period_days, timeframe))
    return resample_ohlcv(df, timeframe)


def _resolve_symbols(args, cfg):
    """Pick the symbol list from --symbols, --watchlist or --universe (in order)."""
    if getattr(args, "symbols", None):
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if getattr(args, "watchlist", False):
        from .watchlist import load

        syms = load()
        if not syms:
            raise SystemExit("watchlist is empty — add symbols with "
                             "'nsetrade watchlist add SYM ...'")
        return syms
    from .universe import get_universe

    universe = getattr(args, "universe", None) or cfg.get("default_universe",
                                                          "nifty50")
    return get_universe(universe)


# --------------------------------------------------------------------------
# analyse
# --------------------------------------------------------------------------


def cmd_analyse(args, cfg):
    from .data import get_provider
    from .patterns import detect_advanced
    from .signals.engine import signal_for_frame

    provider, pconf = _resolve_provider(args, cfg)
    tf = getattr(args, "timeframe", "daily")
    prov = get_provider(provider, pconf)
    df = _fetch(prov, args.symbol, tf, args.days)
    sig = signal_for_frame(args.symbol, df)
    advanced = detect_advanced(df)

    ind = sig.indicators
    print(f"\n=== {sig.symbol}  ({provider}, {tf})  as of {sig.date.date()} ===")
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
    print("  Chart patterns (heuristic — confirm on the chart):")
    if advanced:
        for m in advanced:
            print(f"    • {m.describe()}")
    else:
        print("    • none of the structural patterns detected")

    from .fibonacci import fib_extension, fib_retracement
    fr = fib_retracement(df)
    if fr.found:
        print("  Fibonacci:")
        print(f"    • {fr.describe()}")
        fx = fib_extension(df)
        if fx.found:
            print(f"    • {fx.describe()}")
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

    symbols = _resolve_symbols(args, cfg)

    def progress(done, total, sym):
        print(f"\r  scanning {done}/{total}  {sym:<14}", end="", file=sys.stderr)

    result = screen(
        symbols,
        provider=provider,
        provider_config=pconf,
        period_days=args.days,
        timeframe=getattr(args, "timeframe", "daily"),
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
    tf = getattr(args, "timeframe", "daily")
    prov = get_provider(provider, pconf)
    df = _fetch(prov, args.symbol, tf, args.days)
    suffix = "" if tf == "daily" else f"_{tf}"
    out = args.out or f"{args.symbol}{suffix}.png"
    out = render_chart(f"{args.symbol} ({tf})", df, out_path=out, bars=args.bars)
    print(f"saved chart -> {out}")


# --------------------------------------------------------------------------
# watchlist
# --------------------------------------------------------------------------


def cmd_watchlist(args, cfg):
    from . import watchlist as wl

    action = args.action
    if action == "list":
        syms = wl.load()
        print("\nWatchlist ({} symbols):".format(len(syms)))
        for s in syms:
            print(f"  • {s}")
        print()
    elif action == "add":
        out = wl.add(args.symbols or [])
        print(f"added; watchlist now has {len(out)} symbols")
    elif action == "remove":
        out = wl.remove(args.symbols or [])
        print(f"removed; watchlist now has {len(out)} symbols")
    elif action == "clear":
        wl.clear()
        print("watchlist cleared")
    elif action == "import":
        out = wl.import_csv(args.csv, column=args.column, merge=not args.replace)
        print(f"imported {len(out)} symbols from {args.csv}")
    else:  # pragma: no cover
        raise SystemExit(f"unknown watchlist action {action!r}")


# --------------------------------------------------------------------------
# scan (structural chart patterns across timeframes)
# --------------------------------------------------------------------------


def cmd_scan(args, cfg):
    from .data import get_provider
    from .patterns import detect_advanced

    provider, pconf = _resolve_provider(args, cfg)
    prov = get_provider(provider, pconf)
    symbols = _resolve_symbols(args, cfg)
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    rows = []
    total = len(symbols) * len(timeframes)
    done = 0
    for sym in symbols:
        for tf in timeframes:
            done += 1
            print(f"\r  scanning {done}/{total}  {sym:<14} {tf:<8}", end="",
                  file=sys.stderr)
            try:
                df = _fetch(prov, sym, tf, args.days)
                for m in detect_advanced(df):
                    if args.breakouts_only and m.status != "breakout":
                        continue
                    rows.append((sym, tf, m))
            except Exception:  # noqa: BLE001 - keep scanning
                continue
    print("", file=sys.stderr)

    if not rows:
        print("\nNo structural patterns detected for the given symbols/timeframes.")
        return

    print(f"\nStructural pattern hits ({len(rows)}):")
    print(f"  {'symbol':<12} {'tf':<8} {'pattern':<22} {'dir':<8} "
          f"{'status':<9} {'breakout':>10}")
    print("  " + "-" * 72)
    for sym, tf, m in rows:
        lvl = f"{m.breakout_level:.2f}" if m.breakout_level else "—"
        print(f"  {sym:<12} {tf:<8} {m.name:<22} {m.direction:<8} "
              f"{m.status:<9} {lvl:>10}")
    print("\n  (heuristic detections — always confirm on the chart before acting)\n")


def cmd_edge(args, cfg):
    from .data import get_provider
    from .edge import all_pattern_edges, pattern_edge

    provider, pconf = _resolve_provider(args, cfg)
    prov = get_provider(provider, pconf)
    tf = getattr(args, "timeframe", "daily")
    df = _fetch(prov, args.symbol, tf, args.days)

    print(f"\n=== Pattern edge — {args.symbol} ({tf}) ===")
    print(f"How past breakouts performed over the next {args.forward} bars "
          f"(target {args.target:.0%}):\n")
    if args.pattern:
        edges = [pattern_edge(df, args.pattern, forward_bars=args.forward,
                              target_pct=args.target)]
    else:
        edges = all_pattern_edges(df, forward_bars=args.forward,
                                  target_pct=args.target)
    shown = 0
    for e in sorted(edges, key=lambda x: x.occurrences, reverse=True):
        if e.occurrences == 0:
            continue
        shown += 1
        print(f"  • {e.describe()}")
        print(f"      avg max gain {e.avg_max_favorable:+.1%}, "
              f"avg max drawdown {e.avg_max_adverse:+.1%}")
    if not shown:
        print("  (no historical breakouts of these patterns on this symbol)")
    print("\n  Past performance is descriptive only — small samples are noisy.\n")


def cmd_confluence(args, cfg):
    from .confluence import confluence_scan

    provider, pconf = _resolve_provider(args, cfg)
    symbols = _resolve_symbols(args, cfg)
    tfs = tuple(t.strip() for t in args.timeframes.split(",") if t.strip())

    def progress(done, total, sym):
        print(f"\r  scanning {done}/{total}  {sym:<14}", end="", file=sys.stderr)

    results = confluence_scan(symbols, provider=provider, provider_config=pconf,
                              timeframes=tfs, period_days=args.days,
                              on_progress=progress)
    print("", file=sys.stderr)

    aligned = [c for c in results if c.aligned != "mixed"] if args.aligned_only \
        else results
    top = aligned[:args.top]
    print(f"\nMulti-timeframe confluence (top {len(top)} by conviction):")
    header = f"  {'symbol':<12} {'conviction':>10} {'aligned':<9}  " + "  ".join(
        f"{tf:<7}" for tf in tfs)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in top:
        tf_cells = "  ".join(
            f"{c.per_timeframe[tf].verdict[:7]:<7}" if tf in c.per_timeframe
            else f"{'—':<7}" for tf in tfs)
        print(f"  {c.symbol:<12} {c.conviction:>+10.2f} {c.aligned:<9}  {tf_cells}")
    print()


def cmd_plan(args, cfg):
    from .data import get_provider
    from .patterns import detect_advanced
    from .tradeplan import trade_plan

    provider, pconf = _resolve_provider(args, cfg)
    prov = get_provider(provider, pconf)
    tf = getattr(args, "timeframe", "daily")
    df = _fetch(prov, args.symbol, tf, args.days)

    # use the strongest detected bullish pattern for a measured-move target
    pattern = None
    for m in detect_advanced(df):
        if m.direction == args.direction.replace("long", "bullish").replace(
                "short", "bearish") and m.breakout_level and m.support:
            pattern = m
            break

    plan = trade_plan(args.symbol, df, direction=args.direction,
                      capital=args.capital, risk_pct=args.risk,
                      stop_atr=args.stop_atr, rr_target=args.rr, pattern=pattern)
    print("\n" + plan.describe())


def cmd_fetch_bhavcopy(args, cfg):
    """Download NSE Bhavcopy (bulk EOD, all stocks) into the local store."""
    import datetime as dt

    from .bhavcopy import BhavcopyStore, download_range

    end = dt.date.today()
    start = end - dt.timedelta(days=args.days)
    store = BhavcopyStore(args.db)

    def progress(done, total, d):
        print(f"\r  {d}  ({done}/{total})", end="", file=sys.stderr)

    print(f"Fetching NSE Bhavcopy {start} → {end} into the local store…\n"
          f"(one file per trading day; weekends/holidays skipped)",
          file=sys.stderr)
    try:
        summary = download_range(store, start, end, on_progress=progress)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"\nBhavcopy download failed: {exc}\n"
                         f"Check your internet connection and try again.")
    print("", file=sys.stderr)
    lo, hi = store.date_range()
    print(f"Done. Fetched {summary['trading_days']} trading days; "
          f"store now holds {len(store.symbols())} symbols "
          f"({lo} → {hi}).\n"
          f"Use it with: --provider bhavcopy  (e.g. "
          f"nsetrade opportunities --universe nse_all --provider bhavcopy)")


def cmd_refresh_universe(args, cfg):
    from .universe import refresh_nse_equity_list

    print("Downloading the NSE equity master list…", file=sys.stderr)
    try:
        syms = refresh_nse_equity_list()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Could not refresh the NSE list: {exc}\n"
            f"Check your internet connection and try again.")
    print(f"Cached {len(syms)} NSE symbols. Use them with: "
          f"--universe nse_all")


def cmd_precompute(args, cfg):
    """Precompute an opportunity scan into the local cache (scheduled runs)."""
    from .opportunities import rank_opportunities
    from .scan_cache import ScanCache

    provider, pconf = _resolve_provider(args, cfg)
    universe = args.universe or "nifty50"
    symbols = _resolve_symbols(args, cfg)

    def progress(done, total, sym):
        print(f"\r  scanning {done}/{total}  {sym:<14}", end="", file=sys.stderr)

    result = rank_opportunities(
        symbols, provider=provider, provider_config=pconf,
        period_days=args.days, side=args.side, with_edge=not args.no_edge,
        top=0, on_progress=progress)  # top=0 → keep all, cache the full ranking
    print("", file=sys.stderr)

    rows = [o.as_row() for o in result.opportunities]
    cache = ScanCache(args.db)
    cache.save_run("opportunities", f"{universe}:{args.side}", rows,
                   meta={"provider": provider, "errors": len(result.errors),
                         "scanned": len(symbols)})
    cache.prune(keep_per_key=5)
    print(f"\nCached {len(rows)} ranked {args.side} setups for '{universe}' "
          f"({len(result.errors)} symbols errored).")

    if getattr(args, "with_patterns", False):
        from .pattern_scan import DEFAULT_PATTERNS, scan_for_patterns
        print("Caching pattern picks (cup & handle, darvas box, vcp)…",
              file=sys.stderr)
        hits, _ = scan_for_patterns(
            symbols, DEFAULT_PATTERNS + ["vcp"], provider=provider,
            provider_config=pconf, on_progress=progress)
        print("", file=sys.stderr)
        cache.save_run("patterns", universe, [h.as_row() for h in hits],
                       meta={"patterns": ["Cup & Handle", "Darvas Box", "VCP"]})
        cache.prune(keep_per_key=5)
        print(f"Cached {len(hits)} pattern picks for '{universe}'.")

    print(f"\nView instantly in the dashboard or with:\n"
          f"  nsetrade opportunities --universe {universe} --side {args.side} --cached")


def cmd_opportunities(args, cfg):
    from .opportunities import rank_opportunities

    # Fast path: read a precomputed scan from the cache.
    if getattr(args, "cached", False):
        from datetime import datetime
        from .scan_cache import ScanCache
        cache = ScanCache(getattr(args, "db", None))
        hit = cache.latest("opportunities", f"{args.universe or 'nifty50'}:{args.side}")
        if not hit:
            raise SystemExit(
                "No cached scan found for this universe/side. Run "
                "'nsetrade scan' first (e.g. overnight via Task Scheduler).")
        when = datetime.fromtimestamp(hit["created_at"]).strftime("%Y-%m-%d %H:%M")
        rows = hit["rows"][:args.top]
        print(f"\nTop {len(rows)} {args.side.upper()} opportunities "
              f"(cached scan from {when}):\n")
        _print_table(__import__("pandas").DataFrame(rows))
        return

    provider, pconf = _resolve_provider(args, cfg)
    symbols = _resolve_symbols(args, cfg)

    def progress(done, total, sym):
        print(f"\r  scanning {done}/{total}  {sym:<14}", end="", file=sys.stderr)

    result = rank_opportunities(
        symbols, provider=provider, provider_config=pconf,
        period_days=args.days, side=args.side, with_edge=not args.no_edge,
        top=args.top, on_progress=progress)
    print("", file=sys.stderr)

    df = result.to_frame()
    if df.empty:
        print("No opportunities found.")
        if result.errors:
            print(f"({len(result.errors)} symbols errored; "
                  f"e.g. {next(iter(result.errors))})")
        return

    print(f"\nTop {len(df)} {args.side.upper()} opportunities "
          f"(conviction + historical pattern edge + reward:risk):\n")
    _print_table(df)
    print("\nColumns: score=composite, conviction=multi-timeframe trend, "
          "edge=pattern win-rate/sample, rr=reward:risk.")
    print("Evidence-based ranking — NOT a profit guarantee. Small pattern "
          "samples are noisy; always manage risk.")

    if args.ai:
        from .ai import ThesisConfig, ThesisWriter
        tcfg = ThesisConfig.from_config(cfg)
        if not tcfg.enabled:
            print("\n[--ai needs an Anthropic API key; skipping AI summary]")
            return
        print(f"\nAsking Claude ({tcfg.model}) for a portfolio read…\n",
              file=sys.stderr)
        print("=== AI portfolio read ===\n")
        print(ThesisWriter(tcfg).summarize_opportunities(
            result.opportunities, side=args.side))
        print("\n[AI-generated, grounded in the table above. "
              "Educational only — not investment advice.]")


def cmd_breakouts(args, cfg):
    """Scan a universe for stocks at new N-period highs."""
    from .breakout import PERIODS, scan_breakouts

    provider, pconf = _resolve_provider(args, cfg)
    symbols = _resolve_symbols(args, cfg)
    if args.period not in PERIODS:
        raise SystemExit(f"unknown period {args.period!r}. "
                         f"Choose from: {', '.join(PERIODS)}")

    def progress(done, total, sym):
        print(f"\r  scanning {done}/{total}  {sym:<14}", end="", file=sys.stderr)

    res, errors = scan_breakouts(
        symbols, period=args.period, tol=args.tol / 100.0, provider=provider,
        provider_config=pconf, on_progress=progress)
    print("", file=sys.stderr)
    if not res:
        print(f"No stocks at a new {args.period} high.")
        return
    import pandas as pd
    print(f"\n{len(res)} stocks breaking out to new {args.period} highs:\n")
    _print_table(pd.DataFrame([b.as_row() for b in res]))
    print("\nConfirm trend + volume before acting. Educational only.")


def cmd_picks(args, cfg):
    """Scan a universe for specific chart patterns + their historical edge."""
    from .pattern_scan import DEFAULT_PATTERNS, scan_for_patterns

    provider, pconf = _resolve_provider(args, cfg)
    symbols = _resolve_symbols(args, cfg)
    keys = ([k.strip() for k in args.patterns.split(",") if k.strip()]
            if args.patterns else DEFAULT_PATTERNS)

    def progress(done, total, sym):
        print(f"\r  scanning {done}/{total}  {sym:<14}", end="", file=sys.stderr)

    hits, errors = scan_for_patterns(
        symbols, keys, provider=provider, provider_config=pconf,
        with_edge=not args.no_edge, only_breakouts=args.breakouts_only,
        on_progress=progress)
    print("", file=sys.stderr)

    if not hits:
        print("No matching patterns found.")
        return
    import pandas as pd
    print(f"\nPattern picks ({', '.join(keys)}) — best edge first:\n")
    _print_table(pd.DataFrame([h.as_row() for h in hits]))
    print("\nedge = win-rate/occurrences · robust ✓ = held out-of-sample. "
          "Probability, not a guarantee.")


def cmd_ask(args, cfg):
    """Natural-language screener: describe what you want, Claude builds the filter."""
    from .ai import ThesisConfig
    from .nlscreen import NLScreener

    tcfg = ThesisConfig.from_config(cfg)
    if not tcfg.enabled:
        raise SystemExit(
            "The natural-language screener needs an Anthropic API key "
            "(ai.api_key or ANTHROPIC_API_KEY).")

    provider, pconf = _resolve_provider(args, cfg)
    symbols = _resolve_symbols(args, cfg)

    def progress(done, total, sym):
        print(f"\r  scanning {done}/{total}  {sym:<14}", end="", file=sys.stderr)

    print(f"Interpreting your request with Claude ({tcfg.model})…", file=sys.stderr)
    spec, matches = NLScreener(tcfg).screen(
        args.query, symbols, provider=provider, provider_config=pconf,
        top=args.top, on_progress=progress)
    print("", file=sys.stderr)

    print(f"\nClaude read your request as: {spec.get('explanation', '')}\n")
    if not matches:
        print("No stocks matched that filter.")
        return
    import pandas as pd
    _print_table(pd.DataFrame([o.as_row() for o in matches]))
    print("\nLLM only built the filter; matching is deterministic. "
          "Educational only — not investment advice.")


def cmd_desk(args, cfg):
    """Grade candidates with the multi-agent AI analyst desk."""
    from .ai import ThesisConfig, assemble_context
    from .analyst_desk import AnalystDesk
    from .data import get_provider
    from .resample import resample_ohlcv

    tcfg = ThesisConfig.from_config(cfg)
    if not tcfg.enabled:
        raise SystemExit(
            "The analyst desk needs an Anthropic API key (ai.api_key or "
            "ANTHROPIC_API_KEY). It makes several Claude calls per stock — "
            "run it on a few top candidates, not the whole universe.")

    provider, pconf = _resolve_provider(args, cfg)
    prov = get_provider(provider, pconf)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        # grade the top-N ranked opportunities from a fast local scan first
        from .opportunities import rank_opportunities
        universe = _resolve_symbols(args, cfg)
        print(f"Ranking {len(universe)} stocks to pick the top {args.top}…",
              file=sys.stderr)
        ranked = rank_opportunities(universe, provider=provider,
                                    provider_config=pconf, side=args.side,
                                    top=args.top)
        symbols = [o.symbol for o in ranked.opportunities]
        if not symbols:
            raise SystemExit("No candidates to grade.")

    desk = AnalystDesk(tcfg)
    print(f"\nConvening the AI analyst desk ({tcfg.model}) on "
          f"{len(symbols)} candidate(s)…\n", file=sys.stderr)
    for sym in symbols:
        try:
            daily = prov.history(sym, period_days=args.days)
            frames = {t: resample_ohlcv(daily, t)
                      for t in ("daily", "weekly", "monthly")}
            ctx = assemble_context(sym, daily, with_confluence_frames=frames)
            print(desk.grade(sym, ctx, side=args.side).describe())
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"{sym}: skipped ({exc})")
    print("[AI panel verdicts — educational only, not investment advice. "
          "Grades reflect evidence + scrutiny, not a profit guarantee.]")


def cmd_thesis(args, cfg):
    from .ai import ThesisConfig, ThesisWriter, assemble_context
    from .data import get_provider
    from .resample import resample_ohlcv, scale_period_days

    tcfg = ThesisConfig.from_config(cfg)
    if not tcfg.enabled:
        raise SystemExit(
            "AI thesis needs an Anthropic API key. Set ai.api_key in config.yaml "
            "or export ANTHROPIC_API_KEY. (This sends a numeric summary to "
            "Anthropic — it is the one feature that leaves your machine.)")

    provider, pconf = _resolve_provider(args, cfg)
    prov = get_provider(provider, pconf)
    tf = args.timeframe

    # fetch daily once; build confluence frames if requested
    daily = prov.history(args.symbol, period_days=scale_period_days(args.days, tf))
    df = resample_ohlcv(daily, tf)
    conf_frames = None
    if args.confluence:
        conf_frames = {t: resample_ohlcv(daily, t)
                       for t in ("daily", "weekly", "monthly")}

    ctx = assemble_context(args.symbol, df, timeframe=tf,
                           with_confluence_frames=conf_frames,
                           capital=args.capital, risk_pct=args.risk)
    print(f"\nAsking Claude ({tcfg.model}) for a thesis on {args.symbol}…\n",
          file=sys.stderr)
    writer = ThesisWriter(tcfg)
    print(f"=== AI trade thesis — {args.symbol} ({tf}) ===\n")
    print(writer.write(args.symbol, ctx))
    print("\n[AI-generated, grounded in the numeric summary above. "
          "Educational only — not investment advice.]\n")


def cmd_watch(args, cfg):
    from .alerts import AlertConfig
    from .live import watch

    provider, pconf = _resolve_provider(args, cfg)
    symbols = _resolve_symbols(args, cfg)
    watch(
        symbols,
        provider=provider,
        provider_config=pconf,
        interval_seconds=args.interval,
        only_market_hours=not args.always,
        max_iterations=args.iterations,
        alert_config=AlertConfig.from_config(cfg),
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
    from .patterns import ADVANCED_PATTERNS
    print("\nStructural chart patterns (via 'scan', heuristic):")
    for _, label, b in ADVANCED_PATTERNS:
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
    p.add_argument("--provider", help="data provider: yfinance | bhavcopy "
                                       "(overrides config; goes before the command)")
    sub = p.add_subparsers(dest="command", required=True)

    tf_choices = ["daily", "weekly", "monthly"]

    a = sub.add_parser("analyse", help="analyse one stock")
    a.add_argument("symbol", help="NSE symbol, e.g. RELIANCE")
    a.add_argument("--days", type=int, default=400, help="history window (days)")
    a.add_argument("--timeframe", choices=tf_choices, default="daily",
                   help="candle timeframe")
    a.set_defaults(func=cmd_analyse)

    s = sub.add_parser("screen", help="scan & rank a universe")
    s.add_argument("--universe", help="nifty50 | nifty100 | nifty500")
    s.add_argument("--symbols", help="comma-separated custom symbols")
    s.add_argument("--watchlist", action="store_true",
                   help="use your saved watchlist instead of a universe")
    s.add_argument("--top", type=int, default=10, help="rows per side")
    s.add_argument("--days", type=int, default=400, help="history window (days)")
    s.add_argument("--timeframe", choices=tf_choices, default="daily",
                   help="candle timeframe")
    s.add_argument("--bullish-only", action="store_true",
                   help="hide bearish table")
    s.set_defaults(func=cmd_screen)

    # ---- opportunities (ranked tradeable setups) ----
    op = sub.add_parser("opportunities", aliases=["opps"],
                        help="rank the most tradeable setups across a universe")
    op.add_argument("--universe", help="nifty50 | nifty100 | nifty500")
    op.add_argument("--symbols", help="comma-separated custom symbols")
    op.add_argument("--watchlist", action="store_true",
                    help="use your saved watchlist instead of a universe")
    op.add_argument("--side", choices=["long", "short"], default="long",
                    help="rank long (default) or short setups")
    op.add_argument("--top", type=int, default=15, help="how many to show")
    op.add_argument("--days", type=int, default=500, help="history window (days)")
    op.add_argument("--no-edge", action="store_true",
                    help="skip the historical pattern-edge backtest (faster)")
    op.add_argument("--ai", action="store_true",
                    help="add a Claude portfolio read (needs an API key)")
    op.add_argument("--cached", action="store_true",
                    help="read a precomputed scan instead of recomputing")
    op.add_argument("--db", help="scan cache path (default ~/.nsetrade/scans.db)")
    op.set_defaults(func=cmd_opportunities)

    # ---- refresh-universe (download the full NSE list) ----
    ru = sub.add_parser("refresh-universe",
                        help="download the full NSE equity list (enables --universe nse_all)")
    ru.set_defaults(func=cmd_refresh_universe)

    # ---- fetch-bhavcopy (bulk EOD data for the whole NSE) ----
    fb = sub.add_parser("fetch-bhavcopy",
                        help="download NSE Bhavcopy (all stocks, EOD) to a local store")
    fb.add_argument("--days", type=int, default=400,
                    help="how many calendar days back to fetch (default 400)")
    fb.add_argument("--db", help="store path (default ~/.nsetrade/bhavcopy.db)")
    fb.set_defaults(func=cmd_fetch_bhavcopy)

    # ---- precompute (rank a universe into the cache; for scheduled runs) ----
    pc = sub.add_parser("precompute",
                        help="precompute & cache an opportunity ranking (run nightly)")
    pc.add_argument("--universe", help="nifty50 | nifty100 | nifty500 | nse_all")
    pc.add_argument("--symbols", help="comma-separated custom symbols")
    pc.add_argument("--watchlist", action="store_true",
                    help="scan your saved watchlist")
    pc.add_argument("--side", choices=["long", "short"], default="long")
    pc.add_argument("--days", type=int, default=500, help="history window (days)")
    pc.add_argument("--no-edge", action="store_true",
                    help="skip the pattern-edge backtest (much faster)")
    pc.add_argument("--with-patterns", action="store_true",
                    help="also cache Pattern Picks (cup & handle, darvas, vcp)")
    pc.add_argument("--db", help="scan cache path (default ~/.nsetrade/scans.db)")
    pc.set_defaults(func=cmd_precompute)

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
    c.add_argument("--timeframe", choices=tf_choices, default="daily",
                   help="candle timeframe")
    c.set_defaults(func=cmd_chart)

    # ---- watchlist ----
    wl = sub.add_parser("watchlist", help="manage your personal watchlist")
    wlsub = wl.add_subparsers(dest="action", required=True)
    wlsub.add_parser("list", help="show the watchlist").set_defaults(
        func=cmd_watchlist)
    wa = wlsub.add_parser("add", help="add symbols")
    wa.add_argument("symbols", nargs="+")
    wa.set_defaults(func=cmd_watchlist)
    wr = wlsub.add_parser("remove", help="remove symbols")
    wr.add_argument("symbols", nargs="+")
    wr.set_defaults(func=cmd_watchlist)
    wlsub.add_parser("clear", help="empty the watchlist").set_defaults(
        func=cmd_watchlist)
    wi = wlsub.add_parser("import", help="import symbols from an NSE CSV")
    wi.add_argument("csv", help="path to the NSE CSV (e.g. EQUITY_L.csv)")
    wi.add_argument("--column", help="force a column name (else auto-detect)")
    wi.add_argument("--replace", action="store_true",
                    help="replace the watchlist instead of merging")
    wi.set_defaults(func=cmd_watchlist)

    # ---- scan (structural patterns) ----
    sc = sub.add_parser("scan",
                        help="scan for chart patterns (cup&handle, darvas, flag…)")
    sc.add_argument("--universe", help="nifty50 | nifty100")
    sc.add_argument("--symbols", help="comma-separated custom symbols")
    sc.add_argument("--watchlist", action="store_true",
                    help="scan your saved watchlist")
    sc.add_argument("--timeframes", default="daily,weekly,monthly",
                    help="comma list of daily,weekly,monthly")
    sc.add_argument("--days", type=int, default=400, help="history window (days)")
    sc.add_argument("--breakouts-only", action="store_true",
                    help="only show patterns in breakout (not still forming)")
    sc.set_defaults(func=cmd_scan)

    # ---- edge (pattern historical performance) ----
    ed = sub.add_parser("edge",
                        help="backtest a pattern's historical follow-through")
    ed.add_argument("symbol", help="NSE symbol, e.g. RELIANCE")
    ed.add_argument("--pattern", help="one pattern key (else all are measured)")
    ed.add_argument("--forward", type=int, default=20,
                    help="bars to measure forward return over")
    ed.add_argument("--target", type=float, default=0.05,
                    help="target move fraction (0.05 = 5%%)")
    ed.add_argument("--days", type=int, default=1500, help="history window (days)")
    ed.add_argument("--timeframe", choices=tf_choices, default="daily",
                    help="candle timeframe")
    ed.set_defaults(func=cmd_edge)

    # ---- confluence (multi-timeframe agreement) ----
    cf = sub.add_parser("confluence",
                        help="rank stocks where daily/weekly/monthly agree")
    cf.add_argument("--universe", help="nifty50 | nifty100")
    cf.add_argument("--symbols", help="comma-separated custom symbols")
    cf.add_argument("--watchlist", action="store_true",
                    help="use your saved watchlist")
    cf.add_argument("--timeframes", default="daily,weekly,monthly",
                    help="comma list of timeframes to combine")
    cf.add_argument("--top", type=int, default=15, help="rows to show")
    cf.add_argument("--days", type=int, default=400, help="history window (days)")
    cf.add_argument("--aligned-only", action="store_true",
                    help="only show stocks aligned across all timeframes")
    cf.set_defaults(func=cmd_confluence)

    # ---- plan (auto trade plan) ----
    pl = sub.add_parser("plan", help="generate a trade plan (entry/stop/target/size)")
    pl.add_argument("symbol", help="NSE symbol, e.g. RELIANCE")
    pl.add_argument("--direction", choices=["long", "short"], default="long")
    pl.add_argument("--capital", type=float, default=100_000.0,
                    help="account capital (₹)")
    pl.add_argument("--risk", type=float, default=0.01,
                    help="fraction of capital risked (0.01 = 1%%)")
    pl.add_argument("--stop-atr", type=float, default=2.0,
                    help="stop distance in ATR multiples")
    pl.add_argument("--rr", type=float, default=2.0,
                    help="reward:risk target if no pattern measured move")
    pl.add_argument("--days", type=int, default=400, help="history window (days)")
    pl.add_argument("--timeframe", choices=tf_choices, default="daily",
                    help="candle timeframe")
    pl.set_defaults(func=cmd_plan)

    # ---- breakouts (new N-period high scanner) ----
    bk = sub.add_parser("breakouts",
                        help="find stocks at new highs (3 months/6 months/52 weeks/…)")
    bk.add_argument("--period", default="52 weeks",
                    help="1 month | 3 months | 6 months | 52 weeks | 3 years | All-time")
    bk.add_argument("--universe", help="nifty50 | nifty100 | nifty500 | nse_all")
    bk.add_argument("--symbols", help="comma-separated custom symbols")
    bk.add_argument("--watchlist", action="store_true",
                    help="scan your saved watchlist")
    bk.add_argument("--tol", type=float, default=0.0,
                    help="within %% of the prior high to still count (e.g. 2)")
    bk.set_defaults(func=cmd_breakouts)

    # ---- picks (focused pattern screener) ----
    pk = sub.add_parser("picks",
                        help="scan for specific patterns (cup_and_handle, darvas_box, …)")
    pk.add_argument("--patterns",
                    help="comma-separated detector keys "
                         "(default: cup_and_handle,darvas_box)")
    pk.add_argument("--universe", help="nifty50 | nifty100 | nifty500 | nse_all")
    pk.add_argument("--symbols", help="comma-separated custom symbols")
    pk.add_argument("--watchlist", action="store_true",
                    help="scan your saved watchlist")
    pk.add_argument("--breakouts-only", action="store_true",
                    help="show confirmed breakouts only")
    pk.add_argument("--no-edge", action="store_true",
                    help="skip the historical-edge backtest (faster)")
    pk.set_defaults(func=cmd_picks)

    # ---- ask (natural-language screener) ----
    ak = sub.add_parser("ask",
                        help="natural-language screener: describe what you want")
    ak.add_argument("query", help="e.g. 'bullish weekly cup&handle with 2:1 RR'")
    ak.add_argument("--universe", help="nifty50 | nifty100 | nifty500 | nse_all")
    ak.add_argument("--symbols", help="comma-separated custom symbols")
    ak.add_argument("--watchlist", action="store_true",
                    help="search your saved watchlist")
    ak.add_argument("--top", type=int, default=25, help="max matches to show")
    ak.set_defaults(func=cmd_ask)

    # ---- desk (AI analyst panel grading) ----
    dk = sub.add_parser("desk",
                        help="grade setups with the multi-agent AI analyst panel")
    dk.add_argument("--symbols", help="comma-separated symbols to grade")
    dk.add_argument("--universe", help="grade the top-N from this universe")
    dk.add_argument("--watchlist", action="store_true",
                    help="grade the top-N from your watchlist")
    dk.add_argument("--side", choices=["long", "short"], default="long")
    dk.add_argument("--top", type=int, default=3,
                    help="how many top-ranked candidates to grade (cost control)")
    dk.add_argument("--days", type=int, default=500, help="history window (days)")
    dk.set_defaults(func=cmd_desk)

    # ---- thesis (AI-written) ----
    th = sub.add_parser("thesis",
                        help="AI-written trade thesis (needs an Anthropic API key)")
    th.add_argument("symbol", help="NSE symbol, e.g. RELIANCE")
    th.add_argument("--timeframe", choices=tf_choices, default="daily",
                    help="candle timeframe")
    th.add_argument("--confluence", action="store_true",
                    help="include multi-timeframe confluence in the analysis")
    th.add_argument("--days", type=int, default=500, help="history window (days)")
    th.add_argument("--capital", type=float, default=100_000.0,
                    help="account capital for the trade plan")
    th.add_argument("--risk", type=float, default=0.01,
                    help="fraction of capital risked per trade")
    th.set_defaults(func=cmd_thesis)

    w = sub.add_parser("watch", help="polling watchlist scanner (alerts on signals)")
    w.add_argument("--universe", help="nifty50 | nifty100")
    w.add_argument("--symbols", help="comma-separated watchlist")
    w.add_argument("--watchlist", action="store_true",
                   help="use your saved watchlist")
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
