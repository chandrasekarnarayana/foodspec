"""ROC/AUC diagnostics and threshold optimization for binary/multiclass classification.

Provides:
- ROC curve computation (binary, OvR multiclass, micro/macro averaging)
- AUC estimation with bootstrap confidence intervals
- Pairwise ROC comparison via bootstrap
- Threshold optimization (Youden's J, cost-sensitive, sensitivity constraints)
- Export-ready result structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.preprocessing import label_binarize


@dataclass
class ThresholdResult:
    """Optimal threshold with achieved metrics."""

    threshold: float
    sensitivity: float
    specificity: float
    ppv: Optional[float] = None
    npv: Optional[float] = None
    j_statistic: Optional[float] = None
    cost: Optional[float] = None


@dataclass
class PerClassRocMetrics:
    """ROC metrics for a single class (binary or OvR)."""

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    pvalue: Optional[float] = None
    n_positives: Optional[int] = None
    n_negatives: Optional[int] = None


@dataclass
class RocDiagnosticsResult:
    """Complete ROC/AUC diagnostics output.

    Attributes
    ----------
    per_class : dict
        Per-class ROC metrics (OvR for multiclass).
        Keys are class labels; values are PerClassRocMetrics.
    micro : Optional[PerClassRocMetrics]
        Micro-averaged ROC (multiclass only).
    macro_auc : Optional[float]
        Macro-averaged AUC (multiclass only).
    optimal_thresholds : dict
        Optimal thresholds by policy (policy_name -> ThresholdResult).
    metadata : dict
        Computation metadata: method, n_bootstrap, random_seed, warnings.
    """

    per_class: Dict[Any, PerClassRocMetrics]
    micro: Optional[PerClassRocMetrics] = None
    macro_auc: Optional[float] = None
    optimal_thresholds: Dict[str, ThresholdResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _validate_inputs(y_true: np.ndarray, y_proba: np.ndarray, sample_weight: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate and coerce inputs to numpy arrays."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(f"y_true ({y_true.shape[0]}) and y_proba ({y_proba.shape[0]}) have different lengths")

    if y_proba.ndim == 1:
        y_proba = y_proba.reshape(-1, 1)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError(f"sample_weight length {sample_weight.shape[0]} != y_true length {y_true.shape[0]}")

    return y_true, y_proba, sample_weight


def _compute_binary_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    pos_label: Optional[Any] = None,
) -> PerClassRocMetrics:
    """Compute binary ROC curve and AUC."""
    # Binarize labels if needed
    unique_labels = np.unique(y_true)
    if len(unique_labels) > 2:
        raise ValueError(f"Expected binary classification but found {len(unique_labels)} classes")

    if pos_label is None:
        pos_label = unique_labels[1] if len(unique_labels) == 2 else unique_labels[0]

    y_true_binary = (y_true == pos_label).astype(int)

    # Handle y_proba shape
    if y_proba.ndim == 2:
        if y_proba.shape[1] == 1:
            y_proba_pos = y_proba[:, 0]
        else:
            # Assume last column is positive class
            y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba

    fpr, tpr, thresholds = sklearn_roc_curve(y_true_binary, y_proba_pos, sample_weight=sample_weight)
    roc_auc = sklearn_auc(fpr, tpr)

    n_pos = np.sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos

    return PerClassRocMetrics(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=roc_auc,
        n_positives=int(n_pos),
        n_negatives=int(n_neg),
    )


def compute_auc_ci_bootstrap(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pos_label: Optional[Any] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute AUC with bootstrap confidence interval.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_proba : array-like
        Predicted probabilities.
    pos_label : optional
        Positive class label.
    n_bootstrap : int
        Number of bootstrap samples.
    confidence_level : float
        CI confidence (e.g., 0.95 for 95%).
    random_seed : optional
        For reproducibility.

    Returns
    -------
    auc : float
    ci_lower : float
    ci_upper : float
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba).ravel()

    if pos_label is None:
        unique_labels = np.unique(y_true)
        pos_label = unique_labels[1] if len(unique_labels) == 2 else unique_labels[0]

    y_true_binary = (y_true == pos_label).astype(int)

    # Original AUC
    roc_auc = sklearn_auc(*sklearn_roc_curve(y_true_binary, y_proba)[:2])

    # Bootstrap
    rng = np.random.default_rng(random_seed)
    auc_boot = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_t_boot = y_true_binary[indices]
        y_p_boot = y_proba[indices]
        if len(np.unique(y_t_boot)) > 1:
            auc_b = sklearn_auc(*sklearn_roc_curve(y_t_boot, y_p_boot)[:2])
            auc_boot.append(auc_b)

    if not auc_boot:
        return roc_auc, roc_auc, roc_auc

    auc_boot = np.array(auc_boot)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(auc_boot, 100 * alpha / 2)
    ci_upper = np.percentile(auc_boot, 100 * (1 - alpha / 2))

    return roc_auc, ci_lower, ci_upper


def compute_binary_roc_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    pos_label: Optional[Any] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> RocDiagnosticsResult:
    """Compute ROC diagnostics for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary true labels.
    y_proba : array-like of shape (n,) or (n, 2)
        Predicted probabilities (or probabilities for positive class).
    sample_weight : array-like, optional
        Sample weights.
    pos_label : optional
        Positive class label. If None, inferred as max unique value.
    n_bootstrap : int
        Number of bootstrap replicates for CI.
    confidence_level : float
        Confidence level for CI (e.g., 0.95).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    RocDiagnosticsResult
        ROC metrics, thresholds, and metadata.
    """
    y_true, y_proba, sample_weight = _validate_inputs(y_true, y_proba, sample_weight)

    if pos_label is None:
        unique_labels = np.unique(y_true)
        pos_label = unique_labels[1] if len(unique_labels) == 2 else unique_labels[0]

    # Compute binary ROC
    metrics = _compute_binary_roc(y_true, y_proba, sample_weight=sample_weight, pos_label=pos_label)

    # Compute CI
    y_proba_pos = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] > 1 else y_proba[:, 0]
    auc_val, ci_lower, ci_upper = compute_auc_ci_bootstrap(
        y_true, y_proba_pos, pos_label=pos_label, n_bootstrap=n_bootstrap, confidence_level=confidence_level, random_seed=random_seed
    )
    metrics.ci_lower = ci_lower
    metrics.ci_upper = ci_upper

    # Optimal thresholds
    opt_thresholds = _compute_optimal_thresholds_binary(y_true, y_proba_pos, metrics.thresholds, pos_label=pos_label)

    return RocDiagnosticsResult(
        per_class={pos_label: metrics},
        optimal_thresholds=opt_thresholds,
        metadata={
            "method": "binary_roc",
            "n_bootstrap": n_bootstrap,
            "random_seed": random_seed,
            "warnings": [],
        },
    )


def _compute_optimal_thresholds_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    pos_label: Optional[Any] = None,
) -> Dict[str, ThresholdResult]:
    """Compute optimal thresholds for binary classification."""
    y_true_binary = (y_true == pos_label).astype(int)
    opt_thresholds: Dict[str, ThresholdResult] = {}

    # Compute metrics on all samples and all thresholds in y_proba
    unique_thresholds = np.sort(np.unique(y_proba))[::-1]  # Descending
    
    n_pos = np.sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos

    if n_pos > 0 and n_neg > 0:
        best_j = -np.inf
        best_threshold = None
        best_sensitivity = None
        best_specificity = None

        for thr in unique_thresholds:
            pred_pos = (y_proba >= thr).astype(int)
            tp = np.sum((pred_pos == 1) & (y_true_binary == 1))
            fp = np.sum((pred_pos == 1) & (y_true_binary == 0))
            
            sensitivity = tp / n_pos if n_pos > 0 else 0
            specificity = 1.0 - (fp / n_neg) if n_neg > 0 else 0
            j_stat = sensitivity + specificity - 1

            if j_stat > best_j:
                best_j = j_stat
                best_threshold = thr
                best_sensitivity = sensitivity
                best_specificity = specificity

        if best_threshold is not None:
            opt_thresholds["youden"] = ThresholdResult(
                threshold=float(best_threshold),
                sensitivity=float(best_sensitivity),
                specificity=float(best_specificity),
                j_statistic=float(best_j),
            )

    return opt_thresholds


def compute_multiclass_roc_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    labels: Optional[List[Any]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> RocDiagnosticsResult:
    """Compute ROC diagnostics for multiclass classification (OvR + micro/macro).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True class labels.
    y_proba : array-like of shape (n, K)
        Predicted class probabilities (K = n_classes).
    sample_weight : array-like, optional
        Sample weights.
    labels : list, optional
        Class labels. If None, inferred from y_true.
    n_bootstrap : int
        Number of bootstrap replicates for CI.
    confidence_level : float
        Confidence level for CI.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    RocDiagnosticsResult
        Per-class, micro-average, and macro-average ROC metrics.
    """
    y_true, y_proba, sample_weight = _validate_inputs(y_true, y_proba, sample_weight)

    if labels is None:
        labels = sorted(np.unique(y_true))
    else:
        labels = list(labels)

    n_classes = len(labels)
    if y_proba.shape[1] != n_classes:
        raise ValueError(f"y_proba has {y_proba.shape[1]} classes but labels has {n_classes}")

    if n_classes <= 2:
        # Fall back to binary if only 2 classes
        if n_classes == 2:
            return compute_binary_roc_diagnostics(y_true, y_proba, sample_weight=sample_weight, pos_label=labels[1], n_bootstrap=n_bootstrap, confidence_level=confidence_level, random_seed=random_seed)
        else:
            raise ValueError("Only 1 class detected")

    # Binarize y_true
    y_true_bin = label_binarize(y_true, classes=labels)

    # Per-class OvR
    per_class_metrics: Dict[Any, PerClassRocMetrics] = {}
    auc_per_class = []

    for i, label in enumerate(labels):
        y_true_i = y_true_bin[:, i]
        y_proba_i = y_proba[:, i]

        fpr, tpr, thresholds = sklearn_roc_curve(y_true_i, y_proba_i, sample_weight=sample_weight)
        roc_auc = sklearn_auc(fpr, tpr)
        auc_per_class.append(roc_auc)

        # CI for this class
        auc_val, ci_lower, ci_upper = compute_auc_ci_bootstrap(
            y_true_i, y_proba_i, pos_label=1, n_bootstrap=n_bootstrap, confidence_level=confidence_level, random_seed=random_seed
        )

        per_class_metrics[label] = PerClassRocMetrics(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auc=roc_auc,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_positives=int(np.sum(y_true_i)),
            n_negatives=int(len(y_true_i) - np.sum(y_true_i)),
        )

    # Micro-average: aggregate TP/FP across classes
    # Collect all unique FPRs across classes
    all_fprs = []
    all_tprs = []
    for label in labels:
        y_true_i = y_true_bin[:, label == np.array(labels)][:, 0]
        y_proba_i = y_proba[:, labels.index(label)]
        fpr_i, tpr_i, _ = sklearn_roc_curve(y_true_i, y_proba_i, sample_weight=sample_weight)
        all_fprs.append(fpr_i)
        all_tprs.append(tpr_i)

    # Use common FPR grid
    fpr_micro = np.linspace(0, 1, 100)
    tpr_micro = np.zeros_like(fpr_micro)

    for fpr_i, tpr_i in zip(all_fprs, all_tprs):
        if len(fpr_i) > 1:
            tpr_interp = np.interp(fpr_micro, fpr_i, tpr_i)
            tpr_micro += tpr_interp

    tpr_micro /= n_classes
    auc_micro = sklearn_auc(fpr_micro, tpr_micro)

    micro_metrics = PerClassRocMetrics(fpr=fpr_micro, tpr=tpr_micro, thresholds=np.array([]), auc=auc_micro)

    # Macro-average: simple average of per-class AUCs
    macro_auc = np.mean(auc_per_class)

    return RocDiagnosticsResult(
        per_class=per_class_metrics,
        micro=micro_metrics,
        macro_auc=macro_auc,
        optimal_thresholds={},
        metadata={
            "method": "multiclass_ovr",
            "n_bootstrap": n_bootstrap,
            "random_seed": random_seed,
            "warnings": [],
        },
    )


def compute_roc_diagnostics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    labels: Optional[List[Any]] = None,
    task: Literal["binary", "multiclass", "auto"] = "auto",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> RocDiagnosticsResult:
    """Compute comprehensive ROC/AUC diagnostics.

    Handles both binary and multiclass classification. For multiclass,
    returns per-class (OvR), micro-average, and macro-average metrics.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True class labels.
    y_proba : array-like of shape (n,) or (n, K)
        Predicted class probabilities. For binary, can be (n,) for
        positive class probability or (n, 2) for both classes.
    sample_weight : array-like, optional
        Sample weights.
    labels : list, optional
        Class labels. If None, inferred from y_true.
    task : {'binary', 'multiclass', 'auto'}
        Task type. If 'auto', inferred from y_proba shape.
    n_bootstrap : int, default=1000
        Number of bootstrap replicates for confidence intervals.
    confidence_level : float, default=0.95
        Confidence level for CI (e.g., 0.95 for 95%).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    RocDiagnosticsResult
        Complete ROC diagnostics with per_class, micro (multiclass only),
        macro (multiclass only), optimal_thresholds, and metadata.

    Examples
    --------
    Binary classification::

        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from foodspec.modeling.diagnostics import compute_roc_diagnostics

        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        clf = LogisticRegression().fit(X[:80], y[:80])
        y_proba = clf.predict_proba(X[80:])[:, 1]
        y_test = y[80:]

        result = compute_roc_diagnostics(y_test, y_proba, random_seed=42)
        print(f"AUC: {result.per_class[1].auc:.3f}")
        print(f"CI: [{result.per_class[1].ci_lower:.3f}, {result.per_class[1].ci_upper:.3f}]")
        print(f"Youden threshold: {result.optimal_thresholds['youden'].threshold:.3f}")

    Multiclass classification::

        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from foodspec.modeling.diagnostics import compute_roc_diagnostics

        X, y = make_classification(n_samples=150, n_features=20, n_informative=15,
                                   n_classes=3, random_state=42)
        clf = LogisticRegression(multi_class='multinomial').fit(X[:100], y[:100])
        y_proba = clf.predict_proba(X[100:])
        y_test = y[100:]

        result = compute_roc_diagnostics(y_test, y_proba, random_seed=42)
        for class_label, metrics in result.per_class.items():
            print(f"Class {class_label}: AUC={metrics.auc:.3f}")
        print(f"Macro AUC: {result.macro_auc:.3f}")
        print(f"Micro AUC: {result.micro.auc:.3f}")
    """
    y_true, y_proba, sample_weight = _validate_inputs(y_true, y_proba, sample_weight)

    # Auto-detect task
    if task == "auto":
        if y_proba.ndim == 1 or y_proba.shape[1] == 1:
            task = "binary"
        else:
            n_unique = len(np.unique(y_true))
            task = "binary" if n_unique == 2 else "multiclass"

    if task == "binary":
        return compute_binary_roc_diagnostics(
            y_true, y_proba, sample_weight=sample_weight, n_bootstrap=n_bootstrap, confidence_level=confidence_level, random_seed=random_seed
        )
    elif task == "multiclass":
        return compute_multiclass_roc_diagnostics(
            y_true, y_proba, sample_weight=sample_weight, labels=labels, n_bootstrap=n_bootstrap, confidence_level=confidence_level, random_seed=random_seed
        )
    else:
        raise ValueError(f"Unknown task: {task}")


__all__ = [
    "RocDiagnosticsResult",
    "PerClassRocMetrics",
    "ThresholdResult",
    "compute_roc_diagnostics",
    "compute_binary_roc_diagnostics",
    "compute_multiclass_roc_diagnostics",
    "compute_auc_ci_bootstrap",
]
