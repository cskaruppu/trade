"""nsetrade — NSE stock pattern, signal, screening and backtesting toolkit."""

__version__ = "0.1.0"

from .signals.engine import analyse  # noqa: E402,F401

__all__ = ["analyse", "__version__"]
