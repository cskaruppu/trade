"""Scan and rank a universe of stocks by composite signal score."""

from .screener import screen, ScreenResult

__all__ = ["screen", "ScreenResult"]
