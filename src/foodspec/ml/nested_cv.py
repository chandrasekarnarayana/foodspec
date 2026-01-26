"""Legacy nested CV utilities (deprecated)."""
from __future__ import annotations

from foodspec.utils.deprecation import warn_deprecated_import
from foodspec.modeling.validation.splits import *  # noqa: F401,F403

warn_deprecated_import("foodspec.ml.nested_cv", "foodspec.modeling.validation.splits")

