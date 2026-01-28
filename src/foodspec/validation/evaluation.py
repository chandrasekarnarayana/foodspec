"""Deprecated evaluation shim for foodspec.validation."""

from __future__ import annotations

from foodspec.modeling.evaluation import evaluate_model_cv
from foodspec.utils.deprecation import warn_deprecated_import

warn_deprecated_import("foodspec.validation.evaluation", "foodspec.modeling.evaluation")

__all__ = ["evaluate_model_cv"]
