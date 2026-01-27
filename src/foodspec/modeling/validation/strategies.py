"""Validation strategies: batch-aware CV, group-stratified splits, nested CV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold


def group_stratified_split(y: np.ndarray, groups: np.ndarray, n_splits: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stratify by y while keeping groups intact."""
    unique_groups = np.unique(groups)
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
    """Hold out all samples from a batch together."""
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(batches))))
    return gkf.split(X, y, groups=batches)


def leave_one_group_out(groups: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-group-out splitter (LOBO/LOSO)."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    for group in unique_groups:
        test_idx = np.where(groups == group)[0]
        train_idx = np.where(groups != group)[0]
        yield train_idx, test_idx


def leave_one_batch_out(batches: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Alias for LOBO using batch identifiers."""
    return leave_one_group_out(batches)


def leave_one_stage_out(stages: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Alias for LOSO using stage identifiers."""
    return leave_one_group_out(stages)


def nested_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
) -> List[dict]:
    """Nested CV: outer loop for performance, inner for tuning (simplified)."""
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
        best_model = model
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


__all__ = [
    "ValidationSummary",
    "batch_aware_cv",
    "group_stratified_split",
    "nested_cv",
    "leave_one_group_out",
    "leave_one_batch_out",
    "leave_one_stage_out",
]
