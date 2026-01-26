"""Legacy preprocessing engine (deprecated)."""
from __future__ import annotations

from foodspec.utils.deprecation import warn_deprecated_import
from foodspec.engine.preprocessing.engine import *  # noqa: F401,F403

warn_deprecated_import("foodspec.preprocess.engine", "foodspec.engine.preprocessing.engine")

