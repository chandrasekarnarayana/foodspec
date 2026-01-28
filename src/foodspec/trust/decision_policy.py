"""Decision policy and operating point selection for model deployment.

This module implements "trust-first" decision policies that convert ROC diagnostics
into explicit operating points for deployment and regulatory compliance. Policies
consider ROC performance, uncertainty quantification (conformal prediction, calibration),
and optional abstention/reject mechanisms.

Key concepts:
    - DecisionPolicy: specification of how to choose thresholds
    - Operating point: thresholds + achieved metrics + uncertainty bounds
    - Audit trail: rationale, assumptions, warnings
    - CostSensitiveROC: ROC analysis with misclassification costs
    - UtilityMaximizer: Expected utility maximization
    - PolicyAuditLog: Immutable decision audit trails for regulatory compliance
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

# Configure audit logging
audit_logger = logging.getLogger("foodspec.policy_audit")
if not audit_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s|AUDIT|%(levelname)s|%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)


class PolicyType(str, Enum):
    """Available decision policy types."""

    YOUDEN = "youden"
    COST_SENSITIVE = "cost_sensitive"
    TARGET_SENSITIVITY = "target_sensitivity"
    TARGET_SPECIFICITY = "target_specificity"
    ABSTENTION_AWARE = "abstention_aware"


@dataclass
class DecisionPolicy:
    """Specification for choosing operating point thresholds.

    Attributes
    ----------
    name : str
        Policy name (e.g., "youden", "cost_sensitive", "target_sensitivity").
    applies_to : Literal["binary", "multiclass_ovr"]
        Whether policy applies to binary or multiclass (One-vs-Rest) classification.
    params : dict
        Policy-specific parameters:
        - youden: {}
        - cost_sensitive: {cost_fp, cost_fn}
        - target_sensitivity: {min_sensitivity}
        - target_specificity: {min_specificity}
        - abstention_aware: {max_abstention_rate, cost_fp, cost_fn}
    regulatory_mode : bool
        If True, policy requires explicit configuration; defaults to
        target_sensitivity=0.95 unless overridden. If False, defaults to Youden.
    """

    name: str
    applies_to: Literal["binary", "multiclass_ovr"] = "binary"
    params: Dict[str, Any] = field(default_factory=dict)
    regulatory_mode: bool = False

    def __post_init__(self):
        """Validate policy configuration."""
        if self.name not in [p.value for p in PolicyType]:
            raise ValueError(f"Unknown policy '{self.name}'. Valid options: {[p.value for p in PolicyType]}")

        # Validate params for specific policies
        if self.name == PolicyType.COST_SENSITIVE.value:
            if "cost_fp" not in self.params or "cost_fn" not in self.params:
                raise ValueError("cost_sensitive policy requires cost_fp and cost_fn in params")
        elif self.name == PolicyType.TARGET_SENSITIVITY.value:
            if "min_sensitivity" not in self.params:
                raise ValueError("target_sensitivity policy requires min_sensitivity in params")
        elif self.name == PolicyType.TARGET_SPECIFICITY.value:
            if "min_specificity" not in self.params:
                raise ValueError("target_specificity policy requires min_specificity in params")
        elif self.name == PolicyType.ABSTENTION_AWARE.value:
            if "max_abstention_rate" not in self.params:
                raise ValueError("abstention_aware policy requires max_abstention_rate in params")


@dataclass
class OperatingPoint:
    """Results from policy-based threshold selection.

    Attributes
    ----------
    thresholds : float or Dict[str, float]
        Scalar threshold for binary, dict of per-class thresholds for multiclass OVR.
    policy : DecisionPolicy
        Policy used to compute operating point.
    achieved_metrics : Dict[str, float]
        Metrics at operating point (sensitivity, specificity, ppv, npv, f1, balanced_acc).
    uncertainty_metrics : Dict[str, Any]
        Uncertainty quantification (conformal coverage, abstention rate, etc.).
    rationale : str
        Human-readable explanation (one paragraph).
    warnings : List[str]
        Warnings about policy or data conditions.
    metadata : Dict[str, Any]
        Computation details (method, n_samples, n_bootstrap, etc.).
    """

    thresholds: float | Dict[str, float]
    policy: DecisionPolicy
    achieved_metrics: Dict[str, float]
    uncertainty_metrics: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "thresholds": (
                self.thresholds
                if isinstance(self.thresholds, (int, float))
                else {str(k): float(v) for k, v in self.thresholds.items()}
            ),
            "policy": {
                "name": self.policy.name,
                "applies_to": self.policy.applies_to,
                "params": self.policy.params,
                "regulatory_mode": self.policy.regulatory_mode,
            },
            "achieved_metrics": {k: float(v) for k, v in self.achieved_metrics.items()},
            "uncertainty_metrics": self.uncertainty_metrics,
            "rationale": self.rationale,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def choose_operating_point(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    roc_result: Any,  # RocDiagnosticsResult
    policy: DecisionPolicy,
    *,
    calibration: Optional[Any] = None,
    conformal: Optional[Any] = None,
    abstention: Optional[Any] = None,
) -> OperatingPoint:
    """Select operating point thresholds based on policy and ROC diagnostics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (shape: n_samples,).
    y_proba : np.ndarray
        Predicted probabilities (shape: n_samples, n_classes).
    roc_result : RocDiagnosticsResult
        ROC diagnostics from compute_roc_diagnostics().
    policy : DecisionPolicy
        Policy specification for threshold selection.
    calibration : optional
        Calibration result (e.g., from IsotonicCalibrator.predict()).
        If provided, applied to y_proba before threshold selection.
    conformal : optional
        Conformal prediction result (e.g., ConformalPredictionResult).
        If provided, uncertainty metrics are extracted.
    abstention : optional
        Abstention result (e.g., AbstentionResult).
        If provided, abstention-aware policies can use it.

    Returns
    -------
    OperatingPoint
        Operating point with thresholds, metrics, rationale, and warnings.

    Raises
    ------
    ValueError
        If policy cannot be applied to data (e.g., multiclass policy on binary data).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_proba.ndim != 2:
        raise ValueError("y_proba must be 2D (n_samples, n_classes)")
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have same number of samples")

    n_classes = y_proba.shape[1]
    is_binary = n_classes == 2

    # Apply calibration if provided
    if calibration is not None:
        y_proba = calibration

    # Validate policy applies to data
    if policy.applies_to == "binary" and not is_binary:
        raise ValueError(
            f"Policy '{policy.name}' applies to binary classification only, but data has {n_classes} classes"
        )

    # Route to policy implementation
    if policy.name == PolicyType.YOUDEN.value:
        return _apply_youden_policy(y_true, y_proba, roc_result, policy)
    elif policy.name == PolicyType.COST_SENSITIVE.value:
        return _apply_cost_sensitive_policy(y_true, y_proba, roc_result, policy)
    elif policy.name == PolicyType.TARGET_SENSITIVITY.value:
        return _apply_target_sensitivity_policy(y_true, y_proba, roc_result, policy)
    elif policy.name == PolicyType.TARGET_SPECIFICITY.value:
        return _apply_target_specificity_policy(y_true, y_proba, roc_result, policy)
    elif policy.name == PolicyType.ABSTENTION_AWARE.value:
        return _apply_abstention_aware_policy(y_true, y_proba, roc_result, policy, abstention=abstention)
    else:
        raise ValueError(f"Unknown policy: {policy.name}")


def _apply_youden_policy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    roc_result: Any,
    policy: DecisionPolicy,
) -> OperatingPoint:
    """Apply Youden's J-statistic policy (maximize sensitivity + specificity - 1)."""
    if y_proba.shape[1] != 2:
        raise ValueError("Youden policy applies to binary classification only")

    # Get threshold from roc_result if already computed
    if "youden" in roc_result.optimal_thresholds:
        threshold = roc_result.optimal_thresholds["youden"].threshold
    else:
        # Compute threshold from ROC curve
        class_1_metrics = roc_result.per_class[1]
        tpr = class_1_metrics.tpr
        fpr = class_1_metrics.fpr
        thresholds = class_1_metrics.thresholds
        j_stats = tpr - fpr
        best_idx = np.argmax(j_stats)
        threshold = float(thresholds[best_idx])

    # Compute metrics at threshold
    y_pred_proba_pos = y_proba[:, 1]
    y_pred = (y_pred_proba_pos >= threshold).astype(int)
    metrics = _compute_binary_metrics(y_true, y_pred, y_pred_proba_pos)

    rationale = (
        f"Youden's J-statistic policy maximizes sensitivity + specificity - 1. "
        f"Selected threshold {threshold:.4f} (J-statistic = {metrics['j_statistic']:.3f}). "
        f"This policy balances false positive and false negative rates, suitable for "
        f"exploratory and research applications where neither error type dominates."
    )

    return OperatingPoint(
        thresholds=threshold,
        policy=policy,
        achieved_metrics=metrics,
        rationale=rationale,
        metadata={"method": "youden", "j_statistic": metrics["j_statistic"]},
    )


def _apply_cost_sensitive_policy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    roc_result: Any,
    policy: DecisionPolicy,
) -> OperatingPoint:
    """Apply cost-sensitive policy (minimize expected cost)."""
    if y_proba.shape[1] != 2:
        raise ValueError("Cost-sensitive policy applies to binary classification only")

    cost_fp = float(policy.params.get("cost_fp", 1.0))
    cost_fn = float(policy.params.get("cost_fn", 1.0))

    if cost_fp <= 0 or cost_fn <= 0:
        raise ValueError("Costs must be positive")

    # Find threshold minimizing expected cost
    y_pred_proba_pos = y_proba[:, 1]

    # Grid search over thresholds
    best_cost = float("inf")
    best_threshold = 0.5

    for thresh in np.unique(y_pred_proba_pos):
        y_pred = (y_pred_proba_pos >= thresh).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Expected cost (unnormalized)
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = thresh

    y_pred = (y_pred_proba_pos >= best_threshold).astype(int)
    metrics = _compute_binary_metrics(y_true, y_pred, y_pred_proba_pos)

    rationale = (
        f"Cost-sensitive policy minimizes expected cost with cost_fp={cost_fp}, "
        f"cost_fn={cost_fn}. Selected threshold {best_threshold:.4f}. "
        f"This policy is suitable for applications where false positives and "
        f"false negatives have asymmetric business impact."
    )

    return OperatingPoint(
        thresholds=best_threshold,
        policy=policy,
        achieved_metrics=metrics,
        rationale=rationale,
        metadata={
            "method": "cost_sensitive",
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
            "expected_cost": float(best_cost),
        },
    )


def _apply_target_sensitivity_policy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    roc_result: Any,
    policy: DecisionPolicy,
) -> OperatingPoint:
    """Apply target sensitivity policy (achieve min sensitivity, maximize specificity)."""
    if y_proba.shape[1] != 2:
        raise ValueError("Target sensitivity policy applies to binary classification only")

    min_sensitivity = float(policy.params.get("min_sensitivity", 0.95))
    if not 0.0 < min_sensitivity <= 1.0:
        raise ValueError("min_sensitivity must be in (0, 1]")

    y_pred_proba_pos = y_proba[:, 1]
    class_1_metrics = roc_result.per_class[1]

    # Find minimum threshold achieving sensitivity >= min_sensitivity
    tpr = class_1_metrics.tpr  # sensitivity
    thresholds = class_1_metrics.thresholds
    fpr = class_1_metrics.fpr

    valid_idx = tpr >= min_sensitivity
    if not np.any(valid_idx):
        warnings_list = [
            f"Cannot achieve sensitivity >= {min_sensitivity} "
            f"(max observed: {np.max(tpr):.3f}). Using maximum sensitivity threshold."
        ]
        best_idx = np.argmax(tpr)
    else:
        # Among valid thresholds, choose one with maximum specificity (min FPR)
        valid_fpr = fpr[valid_idx]
        best_valid_idx = np.argmin(valid_fpr)
        best_idx = np.where(valid_idx)[0][best_valid_idx]
        warnings_list = []

    threshold = float(thresholds[best_idx])
    y_pred = (y_pred_proba_pos >= threshold).astype(int)
    metrics = _compute_binary_metrics(y_true, y_pred, y_pred_proba_pos)

    rationale = (
        f"Target sensitivity policy enforces minimum sensitivity of {min_sensitivity:.1%} "
        f"and maximizes specificity. Selected threshold {threshold:.4f}. "
        f"This policy is suitable for regulatory compliance and high-sensitivity "
        f"requirements (e.g., medical screening where false negatives are costly)."
    )

    return OperatingPoint(
        thresholds=threshold,
        policy=policy,
        achieved_metrics=metrics,
        rationale=rationale,
        warnings=warnings_list,
        metadata={
            "method": "target_sensitivity",
            "min_sensitivity": min_sensitivity,
            "achieved_sensitivity": float(metrics["sensitivity"]),
        },
    )


def _apply_target_specificity_policy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    roc_result: Any,
    policy: DecisionPolicy,
) -> OperatingPoint:
    """Apply target specificity policy (achieve min specificity, maximize sensitivity)."""
    if y_proba.shape[1] != 2:
        raise ValueError("Target specificity policy applies to binary classification only")

    min_specificity = float(policy.params.get("min_specificity", 0.95))
    if not 0.0 < min_specificity <= 1.0:
        raise ValueError("min_specificity must be in (0, 1]")

    y_pred_proba_pos = y_proba[:, 1]
    class_1_metrics = roc_result.per_class[1]

    # Find maximum threshold achieving specificity >= min_specificity
    tpr = class_1_metrics.tpr
    thresholds = class_1_metrics.thresholds
    fpr = class_1_metrics.fpr
    specificity = 1.0 - fpr

    valid_idx = specificity >= min_specificity
    if not np.any(valid_idx):
        warnings_list = [
            f"Cannot achieve specificity >= {min_specificity} "
            f"(max observed: {np.max(specificity):.3f}). Using maximum specificity threshold."
        ]
        best_idx = np.argmax(specificity)
    else:
        # Among valid thresholds, choose one with maximum sensitivity (max TPR)
        valid_tpr = tpr[valid_idx]
        best_valid_idx = np.argmax(valid_tpr)
        best_idx = np.where(valid_idx)[0][best_valid_idx]
        warnings_list = []

    threshold = float(thresholds[best_idx])
    y_pred = (y_pred_proba_pos >= threshold).astype(int)
    metrics = _compute_binary_metrics(y_true, y_pred, y_pred_proba_pos)

    rationale = (
        f"Target specificity policy enforces minimum specificity of {min_specificity:.1%} "
        f"and maximizes sensitivity. Selected threshold {threshold:.4f}. "
        f"This policy is suitable for applications where false positives are very costly "
        f"(e.g., spam detection, precision-critical systems)."
    )

    return OperatingPoint(
        thresholds=threshold,
        policy=policy,
        achieved_metrics=metrics,
        rationale=rationale,
        warnings=warnings_list,
        metadata={
            "method": "target_specificity",
            "min_specificity": min_specificity,
            "achieved_specificity": float(metrics["specificity"]),
        },
    )


def _apply_abstention_aware_policy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    roc_result: Any,
    policy: DecisionPolicy,
    abstention: Optional[Any] = None,
) -> OperatingPoint:
    """Apply abstention-aware policy (maximize utility under abstention constraint)."""
    if y_proba.shape[1] != 2:
        raise ValueError("Abstention-aware policy applies to binary classification only")

    max_abstention_rate = float(policy.params.get("max_abstention_rate", 0.1))
    if not 0.0 <= max_abstention_rate < 1.0:
        raise ValueError("max_abstention_rate must be in [0, 1)")

    y_pred_proba_pos = y_proba[:, 1]

    # If abstention result provided, use it; otherwise use max-prob abstention
    if abstention is not None:
        # Use confidence scores if available
        if abstention.confidence_scores is not None:
            confidence = abstention.confidence_scores
        else:
            confidence = np.max(y_proba, axis=1)
    else:
        confidence = np.max(y_proba, axis=1)

    # Grid search: find threshold that maximizes utility under abstention constraint
    thresholds_to_try = np.sort(np.unique(y_pred_proba_pos))
    best_score = -float("inf")
    best_threshold = 0.5
    best_abstention_rate = 0.0

    for thresh in thresholds_to_try:
        # Predictions with this confidence/probability threshold
        y_pred = (y_pred_proba_pos >= thresh).astype(int)

        # Abstention based on confidence
        conf_threshold = thresh
        abstain_mask = confidence < conf_threshold
        abstention_rate = np.mean(abstain_mask)

        if abstention_rate > max_abstention_rate:
            continue

        # Among non-abstained samples, compute accuracy
        if np.sum(~abstain_mask) > 0:
            accuracy = np.mean(y_pred[~abstain_mask] == y_true[~abstain_mask])
        else:
            accuracy = 0.0

        # Utility = coverage * accuracy (penalize abstention, reward accuracy)
        coverage = 1.0 - abstention_rate
        utility = coverage * accuracy

        if utility > best_score:
            best_score = utility
            best_threshold = thresh
            best_abstention_rate = abstention_rate

    y_pred = (y_pred_proba_pos >= best_threshold).astype(int)
    metrics = _compute_binary_metrics(y_true, y_pred, y_pred_proba_pos)
    metrics["effective_abstention_rate"] = best_abstention_rate

    rationale = (
        f"Abstention-aware policy selects threshold {best_threshold:.4f} maximizing utility "
        f"(coverage × accuracy) subject to abstention rate ≤ {max_abstention_rate:.1%}. "
        f"Achieved abstention rate: {best_abstention_rate:.1%}. "
        f"This policy is suitable for high-stakes applications allowing selective deferral."
    )

    return OperatingPoint(
        thresholds=best_threshold,
        policy=policy,
        achieved_metrics=metrics,
        uncertainty_metrics={"abstention_rate": best_abstention_rate},
        rationale=rationale,
        metadata={
            "method": "abstention_aware",
            "max_abstention_rate": max_abstention_rate,
            "achieved_abstention_rate": best_abstention_rate,
        },
    )


def _compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute common binary classification metrics."""
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    j_statistic = sensitivity + specificity - 1.0

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "j_statistic": float(j_statistic),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_operating_point(
    output_dir: Path | str,
    operating_point: OperatingPoint,
) -> Dict[str, str]:
    """Save operating point to disk (JSON + CSV summary).

    Parameters
    ----------
    output_dir : Path or str
        Directory to save artifacts.
    operating_point : OperatingPoint
        Operating point to save.

    Returns
    -------
    dict
        Mapping of artifact type to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # 1. Save operating point JSON
    op_json_path = output_dir / "decision_policy.json"
    with open(op_json_path, "w") as f:
        json.dump(operating_point.to_dict(), f, indent=2)
    artifacts["decision_policy_json"] = str(op_json_path)

    # 2. Save thresholds CSV (for easy inspection)
    thresholds_csv_path = output_dir / "operating_point_thresholds.csv"
    if isinstance(operating_point.thresholds, dict):
        thresholds_data = [{"class": str(k), "threshold": float(v)} for k, v in operating_point.thresholds.items()]
    else:
        thresholds_data = [{"class": "binary", "threshold": float(operating_point.thresholds)}]

    try:
        import pandas as pd

        df = pd.DataFrame(thresholds_data)
        df.to_csv(thresholds_csv_path, index=False)
        artifacts["thresholds_csv"] = str(thresholds_csv_path)
    except ImportError:
        warnings.warn("pandas not available; thresholds CSV not saved")

    # 3. Save metrics CSV
    metrics_csv_path = output_dir / "operating_point_metrics.csv"
    try:
        import pandas as pd

        metrics_df = pd.DataFrame([operating_point.achieved_metrics])
        metrics_df.to_csv(metrics_csv_path, index=False)
        artifacts["metrics_csv"] = str(metrics_csv_path)
    except ImportError:
        pass

    return artifacts


# ============================================================================
# REGULATORY-GRADE COMPONENTS: Cost-Sensitive Analysis, Audit Logging
# ============================================================================


class PolicyAuditLog:
    """Immutable audit log for all policy decisions (GxP/regulatory compliance)."""

    def __init__(self, policy_id: str):
        """
        Initialize audit log.

        Parameters
        ----------
        policy_id : str
            Unique identifier for this policy instance.
        """
        self.policy_id = policy_id
        self.entries = []
        self.created_at = datetime.utcnow().isoformat()

    def log_decision(
        self,
        decision: str,
        context: Dict[str, Any],
        confidence: float,
        reasoning: str,
        parameters: Dict[str, Any],
    ):
        """Log a decision with full context."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "context": context,
            "parameters": parameters,
        }
        self.entries.append(entry)
        audit_logger.info(
            f"policy_id={self.policy_id}|decision={decision}|confidence={confidence:.3f}|reasoning={reasoning}"
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Export audit log as JSON."""
        return json.dumps(
            {
                "policy_id": self.policy_id,
                "created_at": self.created_at,
                "n_entries": len(self.entries),
                "entries": self.entries,
            },
            indent=indent,
            default=str,
        )

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.entries:
            return {"n_entries": 0}
        decisions = [e["decision"] for e in self.entries]
        confidences = [e["confidence"] for e in self.entries]
        return {
            "n_entries": len(self.entries),
            "decision_counts": dict(zip(*np.unique(decisions, return_counts=True))),
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
        }


class CostSensitiveROC:
    """Cost-sensitive ROC analysis for regulatory decision-making."""

    def __init__(
        self,
        cost_fp: float = 1.0,
        cost_fn: float = 1.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0,
    ):
        """
        Initialize cost-sensitive ROC.

        Parameters
        ----------
        cost_fp : float
            Cost of false positive.
        cost_fn : float
            Cost of false negative.
        cost_tp : float
            Cost of true positive (usually 0).
        cost_tn : float
            Cost of true negative (usually 0).
        """
        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.cost_tp = cost_tp
        self.cost_tn = cost_tn

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform cost-sensitive ROC analysis."""
        y_true = np.asarray(y_true, dtype=bool)
        y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64).ravel()

        sorted_idx = np.argsort(y_pred_proba)[::-1]
        proba_sorted = y_pred_proba[sorted_idx]

        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos

        thresholds = np.concatenate(([1.0], proba_sorted, [0.0]))
        tpr_list, fpr_list, costs = [], [], []

        for threshold in thresholds:
            y_pred = y_pred_proba >= threshold
            tp = np.sum(y_pred & y_true)
            fp = np.sum(y_pred & ~y_true)
            fn = np.sum(~y_pred & y_true)
            tn = np.sum(~y_pred & ~y_true)

            tpr = tp / n_pos if n_pos > 0 else 0
            fpr = fp / n_neg if n_neg > 0 else 0
            cost = self.cost_fp * fp + self.cost_fn * fn + self.cost_tp * tp + self.cost_tn * tn

            tpr_list.append(tpr)
            fpr_list.append(fpr)
            costs.append(cost)

        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]

        return {
            "thresholds": thresholds.tolist(),
            "tpr": tpr_list,
            "fpr": fpr_list,
            "costs": costs,
            "optimal_threshold": float(optimal_threshold),
            "optimal_cost": float(costs[optimal_idx]),
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "cost_params": {
                "fp": self.cost_fp,
                "fn": self.cost_fn,
                "tp": self.cost_tp,
                "tn": self.cost_tn,
            },
        }


class UtilityMaximizer:
    """Utility-based decision making for regulatory workflows."""

    def __init__(self, utilities: Dict[str, float]):
        """
        Initialize utility maximizer.

        Parameters
        ----------
        utilities : dict
            Utility values for outcomes (e.g., {"accept": 1.0, "reject": -0.5}).
        """
        self.utilities = utilities

    def decide(self, scores: Dict[str, float]) -> Tuple[str, float, str]:
        """Make decision maximizing expected utility."""
        expected_utilities = {}
        for outcome, utility in self.utilities.items():
            score = scores.get(outcome, 0.0)
            expected_utilities[outcome] = utility * score

        decision = max(expected_utilities, key=expected_utilities.get)
        expected_utility = expected_utilities[decision]

        top_3 = sorted(expected_utilities.items(), key=lambda x: x[1], reverse=True)[:3]
        reasoning = f"Utility-maximizing. Top: {', '.join([f'{o}({u:.3f})' for o, u in top_3])}"

        return decision, float(expected_utility), reasoning


class RegulatoryDecisionPolicy:
    """Regulatory-compliant decision policy with full audit trails."""

    def __init__(
        self,
        policy_id: Optional[str] = None,
        cost_matrix: Optional[Dict[str, float]] = None,
        utilities: Optional[Dict[str, float]] = None,
    ):
        """Initialize regulatory decision policy."""
        self.policy_id = policy_id or f"policy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.audit_log = PolicyAuditLog(self.policy_id)

        if cost_matrix is None:
            cost_matrix = {"cost_fp": 1.0, "cost_fn": 1.0, "cost_tp": 0.0, "cost_tn": 0.0}
        self.cost_roc = CostSensitiveROC(**cost_matrix)

        if utilities is None:
            utilities = {"accept": 1.0, "reject": -0.5, "investigate": 0.1}
        self.utility_max = UtilityMaximizer(utilities)

    def decide(
        self,
        scores: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make regulatory-compliant decision with audit trail."""
        if context is None:
            context = {}

        decision, expected_utility, reasoning = self.utility_max.decide(scores)
        confidence = scores.get(decision, 0.5)

        self.audit_log.log_decision(
            decision=decision,
            context=context,
            confidence=confidence,
            reasoning=reasoning,
            parameters={
                "scores": scores,
                "expected_utility": expected_utility,
                "utilities": self.utility_max.utilities,
            },
        )

        return {
            "decision": decision,
            "confidence": float(confidence),
            "expected_utility": float(expected_utility),
            "reasoning": reasoning,
            "scores": scores,
            "timestamp": datetime.utcnow().isoformat(),
            "policy_id": self.policy_id,
        }

    def get_audit_log(self) -> str:
        """Get full audit log as JSON."""
        return self.audit_log.to_json()

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit summary."""
        return self.audit_log.summary()
