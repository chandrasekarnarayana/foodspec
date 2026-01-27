"""Deprecated metrics shim for foodspec.validation."""
from __future__ import annotations

from foodspec.modeling.validation import metrics as _metrics
from foodspec.modeling.validation.metrics import *  # noqa: F401,F403
from foodspec.utils.deprecation import warn_deprecated_import

warn_deprecated_import("foodspec.validation.metrics", "foodspec.modeling.validation.metrics")

__all__ = list(getattr(_metrics, "__all__", []))
