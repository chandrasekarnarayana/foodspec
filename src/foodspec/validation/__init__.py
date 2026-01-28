"""Legacy validation namespace (deprecated)."""

from __future__ import annotations

from foodspec.modeling.evaluation import evaluate_model_cv
from foodspec.modeling.validation.quality import (
    ValidationError,
    validate_dataset,
    validate_public_evoo_sunflower,
    validate_spectrum_set,
)
from foodspec.modeling.validation.splits import (
    nested_cross_validate,
    nested_cross_validate_custom,
    nested_cross_validate_regression,
)
from foodspec.modeling.validation.strategies import (
    ValidationSummary,
    batch_aware_cv,
    group_stratified_split,
    nested_cv,
)
from foodspec.utils.deprecation import warn_deprecated_import

warn_deprecated_import("foodspec.validation", "foodspec.modeling.validation")

__all__ = [
    "ValidationError",
    "ValidationSummary",
    "validate_dataset",
    "validate_public_evoo_sunflower",
    "validate_spectrum_set",
    "batch_aware_cv",
    "group_stratified_split",
    "nested_cv",
    "nested_cross_validate",
    "nested_cross_validate_custom",
    "nested_cross_validate_regression",
    "evaluate_model_cv",
]
