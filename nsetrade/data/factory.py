"""Build a :class:`DataProvider` from a name + config dict."""

from __future__ import annotations

from typing import Optional

from .base import DataProvider


def get_provider(name: str, config: Optional[dict] = None) -> DataProvider:
    """Return a configured provider instance.

    Parameters
    ----------
    name:
        ``"yfinance"`` or ``"kite"``.
    config:
        The provider-specific options block (e.g. ``config["providers"]["kite"]``).
    """
    name = (name or "yfinance").lower()
    opts = dict(config or {})

    if name in ("yfinance", "yf", "yahoo"):
        from .yfinance_provider import YFinanceProvider

        return YFinanceProvider(**{k: v for k, v in opts.items() if k == "suffix"})

    if name in ("kite", "zerodha"):
        from .kite_provider import KiteProvider

        allowed = {"api_key", "access_token", "exchange"}
        return KiteProvider(**{k: v for k, v in opts.items() if k in allowed})

    raise ValueError(
        f"unknown provider {name!r}. Available: 'yfinance', 'kite'."
    )
