"""Deprecated shim for Ratio-Quality (RQ) engine.

Use foodspec.features.rq instead. This module will be removed in a future release.
"""

from __future__ import annotations

import warnings

from foodspec.features.rq import *  # noqa: F401,F403

warnings.warn(
    "foodspec.rq is deprecated and will be removed in v2.0.0. "
    "Use foodspec.features.rq instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
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
