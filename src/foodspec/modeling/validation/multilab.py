"""Multi-lab validation helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from foodspec.core.errors import FoodSpecValidationError
from foodspec.modeling.validation.metrics import classification_metrics_bundle
from foodspec.modeling.validation.strategies import leave_one_group_out


def leave_one_lab_out(labs: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-lab-out CV splits."""
    labs = np.asarray(labs)
    if labs.size == 0:
        raise FoodSpecValidationError("Lab labels are required for leave-one-lab-out.")
    if len(np.unique(labs)) < 2:
        raise FoodSpecValidationError("Leave-one-lab-out requires at least two labs.")
    return leave_one_group_out(labs)


def multilab_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labs: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute per-lab metrics for a completed validation run."""
    labs = np.asarray(labs)
    if labs.shape[0] != len(y_true):
        raise FoodSpecValidationError("Lab labels length mismatch.")
    report: Dict[str, Any] = {"labs": {}}
    for lab in np.unique(labs):
        mask = labs == lab
        report["labs"][str(lab)] = classification_metrics_bundle(
            y_true[mask], y_pred[mask], y_proba[mask] if y_proba is not None else None
        )
    report["lab_count"] = int(len(report["labs"]))
    return report


__all__ = ["leave_one_lab_out", "multilab_metrics"]
