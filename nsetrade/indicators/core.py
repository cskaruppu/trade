"""Technical indicators implemented with pandas/numpy only.

Each function takes a price Series (or the OHLCV DataFrame) and returns a Series
or DataFrame aligned to the input index. They are deliberately dependency-light
and match the standard textbook formulas (Wilder smoothing for RSI/ATR/ADX).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# --------------------------------------------------------------------------
# Momentum / oscillators
# --------------------------------------------------------------------------


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # when avg_loss == 0 RSI is 100 (pure uptrend)
    out = out.where(avg_loss != 0, 100.0)
    return out


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator %K and %D (0-100)."""
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    rng = (high_max - low_min).replace(0.0, np.nan)
    k = 100 * (df["close"] - low_min) / rng
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing). Useful for stop sizing."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands: middle (SMA), upper, lower and %B."""
    mid = sma(series, period)
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower)
    pct_b = (series - lower) / width.replace(0.0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_pct_b": pct_b}
    )


# --------------------------------------------------------------------------
# Trend strength
# --------------------------------------------------------------------------


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index plus +DI / -DI (Wilder)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / atr_.replace(0.0, np.nan)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        / atr_.replace(0.0, np.nan)
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


# --------------------------------------------------------------------------
# Volume
# --------------------------------------------------------------------------


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).fillna(0.0).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative Volume-Weighted Average Price.

    Note: a true intraday VWAP resets each session. For daily bars this is a
    running anchored VWAP, which is still a useful fair-value reference.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum().replace(0.0, np.nan)
    return (typical * df["volume"]).cumsum() / cum_vol


# --------------------------------------------------------------------------
# Convenience: stack everything onto one frame
# --------------------------------------------------------------------------


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a standard indicator set appended."""
    out = df.copy()
    close = out["close"]
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["sma_200"] = sma(close, 200)
    out["ema_20"] = ema(close, 20)
    out["rsi_14"] = rsi(close, 14)
    out = out.join(macd(close))
    out = out.join(bollinger_bands(close))
    out = out.join(stochastic(out))
    out = out.join(adx(out))
    out["atr_14"] = atr(out, 14)
    out["obv"] = obv(out)
    out["vol_sma_20"] = sma(out["volume"], 20)
    return out
