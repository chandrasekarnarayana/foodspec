"""
Deprecated shim for Ratio-Quality (RQ) engine.

Use foodspec.features.rq instead. This module will be removed in a future release.
"""

from __future__ import annotations

import warnings

from foodspec.features.rq import *  # noqa: F401,F403

warnings.warn(  # noqa: E402
    "foodspec.rq is deprecated; use foodspec.features.rq instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [  # noqa: F405,F401
    "PeakDefinition",
    "RatioDefinition",
    "RQConfig",
    "RatioQualityEngine",
    "RatioQualityResult",
]
