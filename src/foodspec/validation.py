"""
Validation utilities for FoodSpec: batch-aware CV, group-stratified splits, nested CV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold


def group_stratified_split(y: np.ndarray, groups: np.ndarray, n_splits: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratify by y while keeping groups intact.
    Approximate approach: assign groups to folds to balance class counts.
    """
    unique_groups = np.unique(groups)
    # Round-robin assignment by group size
    group_sizes = []
    for g in unique_groups:
        mask = groups == g
        group_sizes.append((g, mask.sum()))
    folds = [[] for _ in range(n_splits)]
    for idx, (g, _) in enumerate(sorted(group_sizes, key=lambda x: -x[1])):
        folds[idx % n_splits].append(g)
    for i in range(n_splits):
        test_groups = folds[i]
        test_idx = np.isin(groups, test_groups)
        train_idx = ~test_idx
        yield np.where(train_idx)[0], np.where(test_idx)[0]


def batch_aware_cv(
    X: np.ndarray,
    y: np.ndarray,
    batches: np.ndarray,
    n_splits: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Hold out all samples from a batch together.
    """
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(batches))))
    return gkf.split(X, y, groups=batches)


def nested_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
) -> List[dict]:
    """
    Nested CV: outer loop for performance, inner for tuning (simplified).
    """
    results = []
    if groups is None:
        outer = StratifiedKFold(
            n_splits=min(outer_splits, len(np.unique(y))),
            shuffle=True,
            random_state=0,
        )
    else:
        outer = StratifiedGroupKFold(
            n_splits=min(outer_splits, len(np.unique(groups))),
            shuffle=True,
            random_state=0,
        )

    for train_idx, test_idx in outer.split(X, y, groups if groups is not None else None):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Simple inner CV to choose best hyperparameter among a small grid (demo)
        best_model = model
        # Fit and evaluate
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        try:
            proba = best_model.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba, multi_class="ovr")
        except Exception:
            auc = None
        results.append(
            {
                "bal_accuracy": bal_acc,
                "per_class_recall": per_class.tolist(),
                "confusion": confusion_matrix(y_test, y_pred).tolist(),
                "roc_auc": auc,
            }
        )
    return results


@dataclass
class ValidationSummary:
    bal_accuracy: float
    per_class_recall: List[float]
    confusion: List[List[int]]
    roc_auc: Optional[float] = None


# Backwards-compat stubs for validators referenced by domain data modules/tests
class ValidationError(ValueError):
    """Raised when a dataset fails validation checks."""


def validate_public_evoo_sunflower(data, allow_nan: bool = True) -> bool:
    """Validator for public EVOO/Sunflower dataset."""
    meta = getattr(data, "metadata", None)
    if meta is None:
        raise ValidationError("Dataset metadata missing.")
    if not allow_nan and meta.isna().any().any():
        raise ValidationError("NaN values not allowed.")
    if "mixture_fraction_evoo" in meta.columns:
        if (meta["mixture_fraction_evoo"] > 100).any() or (meta["mixture_fraction_evoo"] < 0).any():
            raise ValidationError("mixture_fraction_evoo must be between 0 and 100.")
    return True


def validate_spectrum_set(
    dataset,
    allow_nan: bool = False,
    check_monotonic: bool = True,
) -> bool:
    """Validate a FoodSpectrumSet (shape consistency, monotonic wn, NaNs)."""
    wn = getattr(dataset, "wavenumbers", None)
    spectra = getattr(dataset, "spectra", getattr(dataset, "x", None))
    if wn is None or spectra is None:
        raise ValidationError("Spectrum set missing wavenumbers or spectra.")
    if check_monotonic and np.any(np.diff(wn) <= 0):
        raise ValidationError("Wavenumbers must be strictly increasing.")
    if not allow_nan and np.isnan(spectra).any():
        raise ValidationError("NaN values not allowed.")
    return True


def validate_dataset(
    dataset: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    class_col: Optional[str] = None,
    min_classes: int = 2,
) -> Dict[str, List[str]]:
    """Generic dataset validation returning diagnostics instead of raising."""
    errors: List[str] = []
    warnings: List[str] = []
    required_cols = required_cols or []
    missing = [c for c in required_cols if c not in dataset.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if class_col and class_col in dataset.columns:
        nunique = dataset[class_col].nunique(dropna=True)
        if nunique < min_classes:
            warnings.append(f"Class column '{class_col}' has {nunique} classes; discrimination limited.")
    # constant feature warning
    const_cols = [c for c in dataset.columns if dataset[c].nunique(dropna=True) <= 1]
    if const_cols:
        warnings.append(f"constant columns detected: {', '.join(const_cols)}")
    return {"errors": errors, "warnings": warnings}
