"""Textbook pattern reliability — an external sanity benchmark.

These are *reference* figures drawn from widely-cited chart-pattern statistics
(Thomas Bulkowski's *Encyclopedia of Chart Patterns* and similar studies). They
are broad, approximate **market averages across thousands of US-stock examples**
— NOT predictions for any single Indian stock, and NOT precise. EdgeForge shows
them next to its *own* validated, stock-specific edge so a user can sanity-check
whether this particular setup is behaving better or worse than the textbook norm.

Always presented with a caveat: ranges, sourced, "verify". The trusted number is
EdgeForge's own out-of-sample edge — this is only a reference point.
"""

from __future__ import annotations

from typing import Optional

# detector key -> reliability tier + approximate published "target-met" band
PATTERN_BENCHMARKS = {
    "cup_and_handle": {"label": "Cup & Handle", "reliability": "High",
                       "typical_hit": "~50–60%",
                       "source": "Bulkowski — cup with handle, upward breakout"},
    "double_bottom": {"label": "Double Bottom", "reliability": "High",
                      "typical_hit": "~60–70%",
                      "source": "Bulkowski — double bottom (Adam & Adam)"},
    "vcp": {"label": "VCP (Volatility Contraction)", "reliability": "High",
            "typical_hit": "~50–65%",
            "source": "Minervini — volatility contraction / tight bases"},
    "darvas_box": {"label": "Darvas Box", "reliability": "Medium",
                   "typical_hit": "~45–55%",
                   "source": "Darvas box-breakout method"},
    "flat_base": {"label": "Flat Base", "reliability": "High",
                  "typical_hit": "~50–60%",
                  "source": "O'Neil — flat base (shallow shelf after an advance)"},
    "flag": {"label": "Bull Flag", "reliability": "Medium",
             "typical_hit": "~50–60%",
             "source": "Bulkowski — flag, high & tight is strongest"},
    "triangle": {"label": "Ascending Triangle", "reliability": "Medium",
                 "typical_hit": "~55–65%",
                 "source": "Bulkowski — ascending triangle, upward breakout"},
    "double_top": {"label": "Double Top", "reliability": "High",
                   "typical_hit": "~60–70%",
                   "source": "Bulkowski — double top (bearish)"},
    "head_shoulders": {"label": "Head & Shoulders", "reliability": "High",
                       "typical_hit": "~55–65%",
                       "source": "Bulkowski — head-and-shoulders top (bearish)"},
    "wedge": {"label": "Wedge", "reliability": "Low",
              "typical_hit": "~45–55%",
              "source": "Bulkowski — rising/falling wedge"},
    "accumulation": {"label": "Accumulation Base", "reliability": "Medium",
                     "typical_hit": "—",
                     "source": "Wyckoff accumulation (qualitative)"},
}

_CAVEAT = ("Textbook averages from US-market studies (Bulkowski / Minervini), "
           "approximate — a reference, not a forecast. The trusted figure is "
           "EdgeForge's own out-of-sample edge on this stock.")


def benchmark_for(key: Optional[str]) -> Optional[dict]:
    """Return the reference benchmark dict for a detector key, or ``None``."""
    if not key:
        return None
    return PATTERN_BENCHMARKS.get(key)


def benchmark_caveat() -> str:
    return _CAVEAT
