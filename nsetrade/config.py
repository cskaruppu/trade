"""Load optional YAML config (provider credentials, defaults).

Config is entirely optional: with no config.yaml the toolkit defaults to the
free yfinance provider. CLI flags always override config values.
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_CONFIG = {
    "default_provider": "yfinance",
    "default_universe": "nifty50",
    "providers": {"yfinance": {}, "kite": {}},
}


def load_config(path: Optional[str] = None) -> dict:
    """Load config from ``path`` (or ./config.yaml) merged over defaults."""
    cfg = {
        **DEFAULT_CONFIG,
        "providers": dict(DEFAULT_CONFIG["providers"]),
    }
    candidate = path or os.environ.get("NSETRADE_CONFIG") or "config.yaml"
    if os.path.exists(candidate):
        import yaml

        with open(candidate, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        for key, val in loaded.items():
            if key == "providers" and isinstance(val, dict):
                merged = dict(cfg["providers"])
                merged.update(val)
                cfg["providers"] = merged
            else:
                cfg[key] = val
    return cfg


def provider_config(cfg: dict, provider: str) -> dict:
    """Return the options block for a given provider name."""
    return dict((cfg.get("providers") or {}).get(provider, {}) or {})
