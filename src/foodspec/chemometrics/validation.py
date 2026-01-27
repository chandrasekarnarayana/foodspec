"""Legacy chemometrics validation utilities (deprecated)."""
from __future__ import annotations

from foodspec.modeling.validation.metrics import *  # noqa: F401,F403
from foodspec.utils.deprecation import warn_deprecated_import

warn_deprecated_import("foodspec.chemometrics.validation", "foodspec.modeling.validation.metrics")

