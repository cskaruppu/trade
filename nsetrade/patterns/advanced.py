"""Structural chart-pattern detectors: Cup & Handle, Darvas Box, Flag,
Double Bottom/Top and Triangles.

These are *visual* patterns with no single rigorous definition, so each detector
below is a documented heuristic. Treat a hit as a **candidate to confirm on the
chart**, not a guaranteed setup — there will be false positives and misses. Each
detector inspects the most recent formation in a lookback window and returns a
:class:`PatternMatch` describing it (including the breakout level to watch).

All detectors take the canonical OHLCV frame and work on any timeframe (daily,
weekly, monthly) — feed them a resampled frame for higher-timeframe scans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PatternMatch:
    name: str
    found: bool
    direction: str = ""          # "bullish" | "bearish"
    status: str = "none"         # "forming" | "breakout"
    breakout_level: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    note: str = ""
    volume_confirmed: Optional[bool] = None   # set for breakouts in detect_advanced
    overlays: Optional[list] = None           # drawable shapes (curve/line) for charts

    def describe(self) -> str:
        if not self.found:
            return f"{self.name}: not detected"
        lvl = f" breakout>{self.breakout_level:.2f}" if self.breakout_level else ""
        vol = ""
        if self.volume_confirmed is not None:
            vol = " vol✓" if self.volume_confirmed else " vol✗"
        return (f"{self.name} [{self.direction}/{self.status}]{lvl}{vol}"
                + (f" — {self.note}" if self.note else ""))


def volume_confirms(df: pd.DataFrame, *, lookback: int = 50,
                    mult: float = 1.3) -> Optional[bool]:
    """Is the latest bar's volume above ``mult``× the prior ``lookback`` average?

    A real breakout is backed by a surge in participation. Returns ``None`` if
    there isn't enough volume history to judge.
    """
    if "volume" not in df or len(df) < lookback + 1:
        return None
    v = df["volume"].astype(float)
    recent = float(v.iloc[-1])
    avg = float(v.iloc[-lookback - 1:-1].mean())
    if avg <= 0:
        return None
    return recent >= avg * mult


def _na(name: str, note: str = "") -> PatternMatch:
    return PatternMatch(name=name, found=False, note=note)


def _pos_to_dates(index: pd.DatetimeIndex, positions):
    """Map positional indices to actual timestamps (rounded + clamped)."""
    out = []
    for p in positions:
        k = max(0, min(len(index) - 1, int(round(float(p)))))
        out.append(index[k])
    return out


def _quad_curve(index, prices, i0, imid, i1):
    """A smooth parabola through three positional anchors → a drawable curve.

    Used to trace the rounded "U" of a cup. Sampled at each integer bar between
    the rims (exact dates), then drawn with a spline for smoothness.
    """
    xs = np.array([i0, imid, i1], float)
    ys = np.array([prices[i0], prices[imid], prices[i1]], float)
    coeffs = np.polyfit(xs, ys, 2)
    ks = list(range(int(i0), int(i1) + 1))
    return {"kind": "spline",
            "x": [index[k] for k in ks],
            "y": [float(np.polyval(coeffs, k)) for k in ks]}


def _line(index, prices, i0, i1):
    """A straight line between two positional anchors (rim, neckline, …)."""
    return {"kind": "line",
            "x": _pos_to_dates(index, [i0, i1]),
            "y": [float(prices[i0]), float(prices[i1])]}


# --------------------------------------------------------------------------
# Darvas Box
# --------------------------------------------------------------------------


def detect_darvas_box(
    df: pd.DataFrame,
    *,
    box_window: int = 40,
    breakout_window: int = 5,
    max_box_pct: float = 0.20,
) -> PatternMatch:
    """Nicolas Darvas' box: a tight consolidation that price then breaks upward.

    A box is the high/low range over ``box_window`` bars ending ``breakout_window``
    bars ago. It must be reasonably tight (range <= ``max_box_pct``). A hit fires
    when price in the last ``breakout_window`` bars closes above the box top.
    """
    name = "Darvas Box"
    need = box_window + breakout_window + 1
    if len(df) < need:
        return _na(name, f"need >= {need} bars")

    box = df.iloc[-(box_window + breakout_window):-breakout_window]
    recent = df.iloc[-breakout_window:]
    box_top = float(box["high"].max())
    box_bottom = float(box["low"].min())
    if box_bottom <= 0:
        return _na(name)
    box_range = (box_top - box_bottom) / box_bottom
    tight = box_range <= max_box_pct
    broke_up = bool((recent["close"] > box_top).any())
    last = float(df["close"].iloc[-1])

    found = tight and broke_up and last > box_bottom
    status = "breakout" if (found and broke_up) else ("forming" if tight else "none")
    return PatternMatch(
        name=name,
        found=bool(found),
        direction="bullish",
        status=status,
        breakout_level=box_top,
        support=box_bottom,
        resistance=box_top,
        start=box.index[0],
        end=df.index[-1],
        note=f"box {box_bottom:.1f}-{box_top:.1f} ({box_range:.0%} wide)",
    )


# --------------------------------------------------------------------------
# Cup and Handle
# --------------------------------------------------------------------------


def detect_cup_and_handle(
    df: pd.DataFrame,
    *,
    lookback: int = 180,
    rim_tol: float = 0.08,
    min_depth: float = 0.12,
    max_depth: float = 0.50,
    min_cup_bars: int = 20,
    handle_max_retrace: float = 0.5,
) -> PatternMatch:
    """Rounded "U" cup between two similar rims, then a small handle pullback.

    Heuristic: the lowest close sits in the middle of the window (the cup base);
    the highs before and after it (the two rims) are within ``rim_tol`` of each
    other; cup depth is ``min_depth``..``max_depth``; the dip after the right rim
    (the handle) retraces at most ``handle_max_retrace`` of the cup depth. A
    breakout fires when price reaches the rim (resistance).
    """
    name = "Cup & Handle"
    win = df.tail(lookback)
    n = len(win)
    if n < max(40, min_cup_bars + 10):
        return _na(name, "not enough bars")

    close = win["close"].values
    # cup base: lowest close constrained to the middle 50% of the window
    lo, hi = int(n * 0.25), int(n * 0.75)
    b = lo + int(np.argmin(close[lo:hi]))
    if b <= 0 or b >= n - 1:
        return _na(name)

    lr = int(np.argmax(close[:b]))            # left rim
    # right rim = the FIRST local peak after the bottom that recovers to ~the
    # left-rim level (where the handle begins). This stops a post-breakout
    # run-up — price already trading above the rim — from being mistaken for the
    # rim, so the cup is levelled and drawn correctly even after it has broken out.
    left_rim_level = close[lr]
    rim_lo = left_rim_level * (1.0 - rim_tol)
    rr = None
    for i in range(b + 1, n - 1):
        if close[i] >= rim_lo and close[i] >= close[i + 1]:
            rr = i
            break
    if rr is None:                            # no rim-level recovery → still forming
        rr = b + int(np.argmax(close[b:]))
    rim_l, rim_r, bottom = close[lr], close[rr], close[b]
    rim = max(rim_l, rim_r)
    if rim <= 0:
        return _na(name)

    rims_similar = abs(rim_l - rim_r) / rim <= rim_tol
    depth = (rim - bottom) / rim
    depth_ok = min_depth <= depth <= max_depth
    width = rr - lr
    centered = 0.30 <= (b - lr) / max(width, 1) <= 0.70
    wide_enough = width >= min_cup_bars

    # handle = the bars after the right rim
    handle = close[rr:]
    handle_low = float(handle.min()) if len(handle) > 1 else rim_r
    cup_depth_abs = max(rim - bottom, 1e-9)
    handle_retrace = (rim_r - handle_low) / cup_depth_abs
    handle_ok = handle_retrace <= handle_max_retrace

    found = bool(rims_similar and depth_ok and centered and wide_enough and handle_ok)
    last = float(close[-1])
    status = "breakout" if (found and last >= rim * 0.99) else (
        "forming" if found else "none")
    overlays = None
    if found:
        overlays = [
            _quad_curve(win.index, close, lr, b, rr),   # the rounded cup "U"
            _line(win.index, close, lr, rr),            # the rim (resistance) line
        ]
    return PatternMatch(
        name=name,
        found=found,
        direction="bullish",
        status=status,
        breakout_level=float(rim),
        support=float(bottom),
        resistance=float(rim),
        start=win.index[lr],
        end=df.index[-1],
        note=f"depth {depth:.0%}, handle retrace {handle_retrace:.0%}",
        overlays=overlays,
    )


# --------------------------------------------------------------------------
# Flag / Pennant (bull continuation)
# --------------------------------------------------------------------------


def detect_flag(
    df: pd.DataFrame,
    *,
    lookback: int = 45,
    pole_min_gain: float = 0.12,
    pole_max_bars: int = 15,
    flag_min_bars: int = 3,
    flag_max_bars: int = 20,
    max_retrace: float = 0.5,
) -> PatternMatch:
    """Bull flag: a sharp rally (the pole), a shallow drift down/sideways (the
    flag), then a breakout that continues the trend.

    Heuristic: the window's highest close is the pole top; the lowest close in
    the ``pole_max_bars`` before it is the pole base; the pole must gain at least
    ``pole_min_gain``. The bars after the pole top form the flag and must retrace
    no more than ``max_retrace`` of the pole. A breakout fires when price closes
    back above the flag's high.
    """
    name = "Bull Flag"
    win = df.tail(lookback)
    n = len(win)
    if n < flag_min_bars + 8:
        return _na(name, "not enough bars")

    close = win["close"].values
    high = win["high"].values
    low = win["low"].values

    pt = int(np.argmax(close))                # pole top
    if pt < 2 or pt > n - (flag_min_bars + 1):
        return _na(name, "no room for pole/flag")

    start = max(0, pt - pole_max_bars)
    ps = start + int(np.argmin(close[start:pt + 1]))   # pole base
    if close[ps] <= 0:
        return _na(name)
    pole_gain = (close[pt] - close[ps]) / close[ps]
    pole_bars = pt - ps

    flag_close = close[pt + 1:]
    flag_low = float(low[pt + 1:].min())
    flag_bars = len(flag_close)
    # resistance = the flag's upper boundary. Exclude the latest bar (so a
    # breakout bar isn't part of its own wall) and the first post-pole bar
    # (whose wick can still reach back to the pole top).
    flag_highs = high[pt + 1:]
    if len(flag_highs) >= 3:
        flag_high = float(flag_highs[1:-1].max())
    elif len(flag_highs) >= 2:
        flag_high = float(flag_highs[:-1].max())
    else:
        flag_high = float(high[pt])
    cup = max(close[pt] - close[ps], 1e-9)
    retrace = (close[pt] - flag_low) / cup

    pole_ok = pole_gain >= pole_min_gain and 2 <= pole_bars <= pole_max_bars
    flag_ok = flag_min_bars <= flag_bars <= flag_max_bars and retrace <= max_retrace
    last = float(close[-1])

    found = bool(pole_ok and flag_ok)
    status = "breakout" if (found and last >= flag_high) else (
        "forming" if found else "none")
    return PatternMatch(
        name=name,
        found=found,
        direction="bullish",
        status=status,
        breakout_level=flag_high,
        support=flag_low,
        resistance=flag_high,
        start=win.index[ps],
        end=df.index[-1],
        note=f"pole +{pole_gain:.0%} in {pole_bars} bars, flag {flag_bars} bars",
    )


# --------------------------------------------------------------------------
# Swing helpers + Double Bottom/Top + Triangles
# --------------------------------------------------------------------------


def _dedupe_clusters(idxs, values, want_max: bool, min_gap: int = 4):
    """Collapse runs of near-adjacent pivots into one extreme per cluster."""
    if not idxs:
        return []
    groups = [[idxs[0]]]
    for i in idxs[1:]:
        if i - groups[-1][-1] <= min_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    pick = (lambda g: max(g, key=lambda k: values[k])) if want_max else (
        lambda g: min(g, key=lambda k: values[k]))
    return [pick(g) for g in groups]


def _swings(df: pd.DataFrame, left: int = 3, right: int = 3):
    """Return (swing_high_idx, swing_low_idx) as positional index lists.

    Adjacent pivots (e.g. flat tops/plateaus) are collapsed to a single point
    so downstream "rising lows / flat highs" logic sees distinct pivots.
    """
    h, l = df["high"].values, df["low"].values
    n = len(df)
    highs, lows = [], []
    for i in range(left, n - right):
        if h[i] == h[i - left:i + right + 1].max():
            highs.append(i)
        if l[i] == l[i - left:i + right + 1].min():
            lows.append(i)
    highs = _dedupe_clusters(highs, h, want_max=True)
    lows = _dedupe_clusters(lows, l, want_max=False)
    return highs, lows


def detect_double_bottom(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    level_tol: float = 0.04,
    min_separation: int = 8,
) -> PatternMatch:
    """Two swing lows at a similar level ("W") with a peak between; breakout
    above that peak confirms."""
    name = "Double Bottom"
    win = df.tail(lookback)
    if len(win) < 30:
        return _na(name, "not enough bars")
    _, lows = _swings(win)
    if len(lows) < 2:
        return _na(name)

    low_vals = win["low"].values
    high_vals = win["high"].values
    close = float(win["close"].iloc[-1])
    # scan recent pairs of swing lows
    for a in range(len(lows) - 1):
        for b in range(len(lows) - 1, a, -1):
            i, j = lows[a], lows[b]
            if j - i < min_separation:
                continue
            la, lb = low_vals[i], low_vals[j]
            if min(la, lb) <= 0:
                continue
            if abs(la - lb) / min(la, lb) > level_tol:
                continue
            peak = float(high_vals[i:j + 1].max())
            if peak <= max(la, lb):
                continue
            status = "breakout" if close >= peak * 0.99 else "forming"
            peak_idx = i + int(np.argmax(high_vals[i:j + 1]))
            overlays = [
                {"kind": "line",                       # the W: low → peak → low
                 "x": _pos_to_dates(win.index, [i, peak_idx, j]),
                 "y": [float(la), float(peak), float(lb)]},
                {"kind": "line",                       # neckline (breakout level)
                 "x": _pos_to_dates(win.index, [i, len(win) - 1]),
                 "y": [float(peak), float(peak)]},
            ]
            return PatternMatch(
                name=name, found=True, direction="bullish", status=status,
                breakout_level=peak, support=float(min(la, lb)), resistance=peak,
                start=win.index[i], end=df.index[-1],
                note=f"bottoms ~{(la + lb) / 2:.1f}, neckline {peak:.1f}",
                overlays=overlays,
            )
    return _na(name)


def detect_double_top(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    level_tol: float = 0.04,
    min_separation: int = 8,
) -> PatternMatch:
    """Two swing highs at a similar level ("M") with a trough between; breakdown
    below that trough confirms."""
    name = "Double Top"
    win = df.tail(lookback)
    if len(win) < 30:
        return _na(name, "not enough bars")
    highs, _ = _swings(win)
    if len(highs) < 2:
        return _na(name)

    high_vals = win["high"].values
    low_vals = win["low"].values
    close = float(win["close"].iloc[-1])
    for a in range(len(highs) - 1):
        for b in range(len(highs) - 1, a, -1):
            i, j = highs[a], highs[b]
            if j - i < min_separation:
                continue
            ha, hb = high_vals[i], high_vals[j]
            if min(ha, hb) <= 0:
                continue
            if abs(ha - hb) / min(ha, hb) > level_tol:
                continue
            trough = float(low_vals[i:j + 1].min())
            if trough >= min(ha, hb):
                continue
            status = "breakout" if close <= trough * 1.01 else "forming"
            return PatternMatch(
                name=name, found=True, direction="bearish", status=status,
                breakout_level=trough, support=trough, resistance=float(max(ha, hb)),
                start=win.index[i], end=df.index[-1],
                note=f"tops ~{(ha + hb) / 2:.1f}, neckline {trough:.1f}",
            )
    return _na(name)


def detect_triangle(
    df: pd.DataFrame,
    *,
    lookback: int = 80,
    flat_tol: float = 0.03,
    min_pivots: int = 2,
) -> PatternMatch:
    """Ascending triangle (flat highs, rising lows) or descending triangle
    (flat lows, falling highs)."""
    name = "Triangle"
    win = df.tail(lookback)
    if len(win) < 30:
        return _na(name, "not enough bars")
    highs, lows = _swings(win)
    if len(highs) < min_pivots or len(lows) < min_pivots:
        return _na(name)

    hv = win["high"].values
    lv = win["low"].values
    sh = [hv[i] for i in highs[-3:]]
    sl = [lv[i] for i in lows[-3:]]
    close = float(win["close"].iloc[-1])

    def _flat(vals):
        return (max(vals) - min(vals)) / min(vals) <= flat_tol if min(vals) > 0 else False

    def _rising(vals):
        return all(vals[k] < vals[k + 1] for k in range(len(vals) - 1))

    def _falling(vals):
        return all(vals[k] > vals[k + 1] for k in range(len(vals) - 1))

    # ascending: flat resistance + rising support -> bullish
    if _flat(sh) and _rising(sl):
        res = float(np.mean(sh))
        status = "breakout" if close >= res * 0.99 else "forming"
        return PatternMatch(
            name="Ascending Triangle", found=True, direction="bullish",
            status=status, breakout_level=res, support=float(sl[0]), resistance=res,
            start=win.index[lows[-3] if len(lows) >= 3 else lows[0]],
            end=df.index[-1], note="flat highs, rising lows",
        )
    # descending: flat support + falling highs -> bearish
    if _flat(sl) and _falling(sh):
        sup = float(np.mean(sl))
        status = "breakout" if close <= sup * 1.01 else "forming"
        return PatternMatch(
            name="Descending Triangle", found=True, direction="bearish",
            status=status, breakout_level=sup, support=sup, resistance=float(sh[0]),
            start=win.index[highs[-3] if len(highs) >= 3 else highs[0]],
            end=df.index[-1], note="flat lows, falling highs",
        )
    return _na(name)


# --------------------------------------------------------------------------
# Head & Shoulders (bearish) + Inverse H&S (bullish)
# --------------------------------------------------------------------------


def detect_head_shoulders(
    df: pd.DataFrame,
    *,
    lookback: int = 160,
    shoulder_tol: float = 0.06,
    min_separation: int = 4,
) -> PatternMatch:
    """Three peaks with a higher middle (head) and similar shoulders, broken at
    the neckline (the two troughs) → bearish. The mirror image (three troughs,
    lower middle) is the bullish Inverse Head & Shoulders.
    """
    name = "Head & Shoulders"
    win = df.tail(lookback)
    if len(win) < 40:
        return _na(name, "not enough bars")
    highs, lows = _swings(win)
    hv, lv = win["high"].values, win["low"].values
    close = float(win["close"].iloc[-1])

    # bearish: last three swing highs, middle highest, shoulders similar
    if len(highs) >= 3:
        ls, hd, rs = highs[-3], highs[-2], highs[-1]
        if rs - ls >= 2 * min_separation:
            lsh, hdh, rsh = hv[ls], hv[hd], hv[rs]
            if (hdh > lsh and hdh > rsh and min(lsh, rsh) > 0
                    and abs(lsh - rsh) / min(lsh, rsh) <= shoulder_tol):
                t1 = float(lv[ls:hd + 1].min())
                t2 = float(lv[hd:rs + 1].min())
                neck = (t1 + t2) / 2.0
                if 0 < neck < min(lsh, rsh):
                    status = "breakout" if close <= neck * 1.01 else "forming"
                    overlays = [
                        {"kind": "line",               # shoulders + head outline
                         "x": _pos_to_dates(win.index, [ls, hd, rs]),
                         "y": [float(lsh), float(hdh), float(rsh)]},
                        {"kind": "line",               # neckline
                         "x": _pos_to_dates(win.index, [ls, len(win) - 1]),
                         "y": [float(neck), float(neck)]},
                    ]
                    return PatternMatch(
                        name=name, found=True, direction="bearish", status=status,
                        breakout_level=neck, support=neck, resistance=float(hdh),
                        start=win.index[ls], end=df.index[-1],
                        note=f"head {hdh:.1f}, neckline {neck:.1f}", overlays=overlays)

    # bullish inverse: last three swing lows, middle lowest
    if len(lows) >= 3:
        ls, hd, rs = lows[-3], lows[-2], lows[-1]
        if rs - ls >= 2 * min_separation:
            lsl, hdl, rsl = lv[ls], lv[hd], lv[rs]
            if (hdl < lsl and hdl < rsl and min(lsl, rsl) > 0
                    and abs(lsl - rsl) / min(lsl, rsl) <= shoulder_tol):
                p1 = float(hv[ls:hd + 1].max())
                p2 = float(hv[hd:rs + 1].max())
                neck = (p1 + p2) / 2.0
                if neck > max(lsl, rsl):
                    status = "breakout" if close >= neck * 0.99 else "forming"
                    return PatternMatch(
                        name="Inverse Head & Shoulders", found=True,
                        direction="bullish", status=status, breakout_level=neck,
                        support=float(hdl), resistance=neck,
                        start=win.index[ls], end=df.index[-1],
                        note=f"head {hdl:.1f}, neckline {neck:.1f}")
    return _na(name)


# --------------------------------------------------------------------------
# Wedge (rising = bearish, falling = bullish)
# --------------------------------------------------------------------------


def detect_wedge(
    df: pd.DataFrame,
    *,
    lookback: int = 90,
    conv_ratio: float = 0.7,
) -> PatternMatch:
    """Two converging trendlines. Rising wedge (both lines sloping up, gap
    narrowing) breaks down → bearish; falling wedge (both sloping down) breaks
    up → bullish.
    """
    name = "Wedge"
    win = df.tail(lookback)
    if len(win) < 30:
        return _na(name, "not enough bars")
    highs, lows = _swings(win)
    if len(highs) < 3 or len(lows) < 3:
        return _na(name)
    hv, lv = win["high"].values, win["low"].values
    hi, lo = highs[-3:], lows[-3:]
    sh = [float(hv[i]) for i in hi]
    sl = [float(lv[i]) for i in lo]
    close = float(win["close"].iloc[-1])

    def _slope(idx, val):
        return (val[-1] - val[0]) / max(idx[-1] - idx[0], 1)

    sh_sl, sl_sl = _slope(hi, sh), _slope(lo, sl)
    gap_start = sh[0] - sl[0]
    gap_end = sh[-1] - sl[-1]
    converging = 0 < gap_end < conv_ratio * gap_start

    if converging and sh_sl > 0 and sl_sl > 0:  # rising wedge → bearish
        line = sl[-1]
        status = "breakout" if close <= line else "forming"
        return PatternMatch(
            name="Rising Wedge", found=True, direction="bearish", status=status,
            breakout_level=float(line), support=float(line), resistance=float(sh[-1]),
            start=win.index[hi[0]], end=df.index[-1], note="converging, sloping up")
    if converging and sh_sl < 0 and sl_sl < 0:  # falling wedge → bullish
        line = sh[-1]
        status = "breakout" if close >= line else "forming"
        return PatternMatch(
            name="Falling Wedge", found=True, direction="bullish", status=status,
            breakout_level=float(line), support=float(sl[-1]), resistance=float(line),
            start=win.index[lo[0]], end=df.index[-1], note="converging, sloping down")
    return _na(name)


# --------------------------------------------------------------------------
# VCP — Volatility Contraction Pattern (Minervini)
# --------------------------------------------------------------------------


def detect_vcp(
    df: pd.DataFrame,
    *,
    lookback: int = 160,
    min_contractions: int = 2,
    max_first_depth: float = 0.45,
    max_last_depth: float = 0.15,
) -> PatternMatch:
    """Volatility Contraction Pattern: a series of progressively *tighter*
    pullbacks as the stock coils near its highs, then a breakout above the
    final pivot.

    Heuristic: walk the swing pivots and measure each high→low pullback depth.
    The recent pullbacks must contract (each shallower than the last), the first
    not too deep and the last tight; the pivot is the most recent swing high and
    a breakout fires when price reaches it.
    """
    name = "VCP"
    win = df.tail(lookback)
    if len(win) < 40:
        return _na(name, "not enough bars")
    highs, lows = _swings(win, left=3, right=3)
    if len(highs) < 2 or len(lows) < 2:
        return _na(name)
    hv, lv = win["high"].values, win["low"].values

    pivots = sorted([(i, "H") for i in highs] + [(i, "L") for i in lows])
    contractions = []           # (high_idx, low_idx, depth)
    last_high = None
    for idx, kind in pivots:
        if kind == "H":
            last_high = (idx, hv[idx])
        elif last_high is not None:
            hi = last_high[1]
            low = lv[idx]
            if hi > 0 and low < hi:
                contractions.append((last_high[0], idx, (hi - low) / hi))
            last_high = None
    if len(contractions) < min_contractions:
        return _na(name, "too few contractions")

    take = 3 if len(contractions) >= 3 else min_contractions
    recent = contractions[-take:]
    depths = [c[2] for c in recent]
    contracting = all(depths[k + 1] < depths[k] for k in range(len(depths) - 1))
    if not (contracting and depths[0] <= max_first_depth
            and depths[-1] <= max_last_depth):
        return _na(name)

    pivot = float(hv[highs[-1]])
    close = float(win["close"].iloc[-1])
    status = "breakout" if close >= pivot * 0.99 else "forming"
    return PatternMatch(
        name=name, found=True, direction="bullish", status=status,
        breakout_level=pivot, support=float(lv[recent[-1][1]]), resistance=pivot,
        start=win.index[recent[0][0]], end=df.index[-1],
        note=f"{len(recent)} contractions "
             f"{', '.join(f'{d:.0%}' for d in depths)}")


# --------------------------------------------------------------------------
# Run them all
# --------------------------------------------------------------------------

ADVANCED_DETECTORS = {
    "cup_and_handle": detect_cup_and_handle,
    "darvas_box": detect_darvas_box,
    "flag": detect_flag,
    "double_bottom": detect_double_bottom,
    "double_top": detect_double_top,
    "triangle": detect_triangle,
    "head_shoulders": detect_head_shoulders,
    "wedge": detect_wedge,
    "vcp": detect_vcp,
}

ADVANCED_PATTERNS = [
    ("cup_and_handle", "Cup & Handle", 1),
    ("darvas_box", "Darvas Box", 1),
    ("flag", "Bull Flag", 1),
    ("double_bottom", "Double Bottom", 1),
    ("double_top", "Double Top", -1),
    ("triangle", "Triangle (asc/desc)", 0),
    ("head_shoulders", "Head & Shoulders (+inverse)", 0),
    ("wedge", "Wedge (rising/falling)", 0),
    ("vcp", "VCP (Volatility Contraction)", 1),
]


def detect_advanced(df: pd.DataFrame, only_found: bool = True) -> list[PatternMatch]:
    """Run every advanced detector and return the matches.

    With ``only_found=True`` (default) only detected patterns are returned.
    """
    results = []
    for fn in ADVANCED_DETECTORS.values():
        try:
            m = fn(df)
        except Exception:  # noqa: BLE001 - a detector should never break a scan
            continue
        if m.found and m.status == "breakout":
            m.volume_confirmed = volume_confirms(df)
        if m.found or not only_found:
            results.append(m)
    return results
