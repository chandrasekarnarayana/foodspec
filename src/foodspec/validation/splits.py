"""Deprecated splits shim for foodspec.validation."""

from __future__ import annotations

from foodspec.modeling.validation.splits import (
    nested_cross_validate,
    nested_cross_validate_custom,
    nested_cross_validate_regression,
)
from foodspec.utils.deprecation import warn_deprecated_import

warn_deprecated_import("foodspec.validation.splits", "foodspec.modeling.validation.splits")

__all__ = [
    "nested_cross_validate",
    "nested_cross_validate_custom",
    "nested_cross_validate_regression",
]
