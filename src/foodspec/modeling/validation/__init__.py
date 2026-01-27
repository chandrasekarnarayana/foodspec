"""Validation helpers for modeling."""
from __future__ import annotations

from .metrics import compute_classification_metrics, compute_regression_metrics
from .multilab import leave_one_lab_out, multilab_metrics
from .quality import (
    ValidationError,
    validate_dataset,
    validate_public_evoo_sunflower,
    validate_spectrum_set,
)
from .splits import (
    nested_cross_validate,
    nested_cross_validate_custom,
    nested_cross_validate_regression,
)
from .strategies import (
    ValidationSummary,
    batch_aware_cv,
    group_stratified_split,
    leave_one_batch_out,
    leave_one_group_out,
    leave_one_stage_out,
    nested_cv,
)

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "ValidationError",
    "validate_dataset",
    "validate_public_evoo_sunflower",
    "validate_spectrum_set",
    "ValidationSummary",
    "batch_aware_cv",
    "group_stratified_split",
    "nested_cv",
    "leave_one_group_out",
    "leave_one_batch_out",
    "leave_one_stage_out",
    "leave_one_lab_out",
    "multilab_metrics",
    "nested_cross_validate",
    "nested_cross_validate_custom",
    "nested_cross_validate_regression",
]
