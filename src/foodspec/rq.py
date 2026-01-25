from __future__ import annotations
"""
Deprecated shim for Ratio-Quality (RQ) engine.

Use foodspec.features.rq instead. This module will be removed in a future release.
"""

"""
rq - DEPRECATED

.. deprecated:: 1.1.0
    This module is deprecated and will be removed in v2.0.0.
    Use foodspec.features.rq instead.

This module is maintained for backward compatibility only.
All new code should use the modern API.

Migration Guide:
    Old: from foodspec.rq import ...
    New: from foodspec.features.rq import ...

See: docs/migration/v1-to-v2.md
"""

import warnings

warnings.warn(
    f"foodspec.rq is deprecated and will be removed in v2.0.0. "
    f"Use foodspec.features.rq instead. "
    f"See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Original module content continues below...
# ==============================================





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
