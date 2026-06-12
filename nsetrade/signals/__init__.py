"""Combine indicators and patterns into a single scored signal."""

from .engine import analyse, signal_for_frame, Signal

__all__ = ["analyse", "signal_for_frame", "Signal"]
