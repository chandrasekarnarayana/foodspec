"""Multivariate QC: outliers, Hotelling T², batch drift."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ReasonCode = Literal["MV_OUTLIER_T2", "MV_OUTLIER_OD", "MV_DRIFT_CENTROID"]


@dataclass
class MultivariateQCPolicy:
    """Policy describing multivariate QC actions and thresholds."""

    action: Literal["flag", "drop", "down_weight"] = "flag"
    severity: Literal["warn", "fail"] = "warn"
    threshold_strategy: Literal["quantile", "chi2", "mad"] = "chi2"
    alpha: float = 0.01
    quantile: float = 0.99
    mad_multiplier: float = 4.5

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "MultivariateQCPolicy":
        if payload is None:
            return cls()
        return cls(
            action=payload.get("action", payload.get("policy", "flag")),
            severity=payload.get("severity", "warn"),
            threshold_strategy=payload.get("threshold_strategy", payload.get("strategy", "chi2")),
            alpha=float(payload.get("alpha", 0.01)),
            quantile=float(payload.get("quantile", 0.99)),
            mad_multiplier=float(payload.get("mad_multiplier", 4.5)),
        )


def _coerce_scores(scores: Any) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2:
        raise ValueError("scores must be a 2D array")
    return arr


def summarize_scores(scores: Any) -> dict:
    arr = _coerce_scores(scores)
    return {"n_samples": int(arr.shape[0]), "n_components": int(arr.shape[1])}


def compute_pca_outlier_scores(
    X: Any,
    method: Literal["robust", "classic"] = "robust",
) -> Dict[str, Any]:
    """Compute score distance and orthogonal distance from PCA space.

    Returns distances and method-dependent thresholds for downstream flagging.
    """

    X_arr = _coerce_scores(X)
    # Center for determinism
    X_centered = X_arr - np.mean(X_arr, axis=0)
    u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    scores = u * s
    recon = scores @ vt
    residual = X_centered - recon
    od = np.linalg.norm(residual, axis=1)

    if method == "classic":
        cov = np.cov(scores, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        sd = np.einsum("ij,jk,ik->i", scores, cov_inv, scores)
        chi_thr = float(stats.chi2.ppf(0.975, df=scores.shape[1]))
        od_thr = float(np.percentile(od, 97.5))
    else:
        med = np.median(scores, axis=0)
        mad = np.median(np.abs(scores - med), axis=0) + 1e-9
        sd = np.sum(((scores - med) / mad) ** 2, axis=1)
        chi_thr = float(np.median(sd) + 4.5 * stats.median_abs_deviation(sd, scale=1.0))
        od_thr = float(np.median(od) + 4.5 * stats.median_abs_deviation(od, scale=1.0))

    thresholds = {"score_distance": chi_thr, "orthogonal_distance": od_thr}
    return {
        "scores": scores,
        "score_distance": sd,
        "orthogonal_distance": od,
        "thresholds": thresholds,
        "method": method,
    }


def hotelling_t2(
    Z: Any,
    covariance: Literal["empirical", "robust"] = "empirical",
    alpha: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """Compute Hotelling's T² statistic and control limit."""

    scores = _coerce_scores(Z)
    mu = np.mean(scores, axis=0)
    if covariance == "robust":
        med = np.median(scores, axis=0)
        mad = np.median(np.abs(scores - med), axis=0) + 1e-9
        normed = (scores - med) / mad
        cov = np.cov(normed, rowvar=False)
    else:
        cov = np.cov(scores, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    diffs = scores - mu
    t2 = np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)
    limit = float(stats.chi2.ppf(1 - alpha, df=scores.shape[1]))
    return t2, limit


def outlier_flags(
    sample_ids: Sequence[Any],
    scores: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    policy: MultivariateQCPolicy,
) -> pd.DataFrame:
    """Assemble outlier flags with reason codes and actions."""

    n = len(sample_ids)
    sd = scores.get("score_distance")
    od = scores.get("orthogonal_distance")
    t2 = scores.get("t2")

    rows = []
    for i in range(n):
        sd_i = float(sd[i]) if sd is not None else None
        od_i = float(od[i]) if od is not None else None
        t2_i = float(t2[i]) if t2 is not None else None

        reason: Optional[ReasonCode] = None
        score_val: Optional[float] = None
        threshold_val: Optional[float] = None

        if t2 is not None and thresholds.get("t2") is not None and t2_i is not None:
            threshold_val = thresholds["t2"]
            if t2_i > threshold_val:
                reason = "MV_OUTLIER_T2"
                score_val = t2_i
        if reason is None and sd is not None:
            threshold_val = thresholds.get("score_distance")
            if threshold_val is not None and sd_i is not None and sd_i > threshold_val:
                reason = "MV_OUTLIER_T2"
                score_val = sd_i
        if reason is None and od is not None:
            threshold_val = thresholds.get("orthogonal_distance")
            if threshold_val is not None and od_i is not None and od_i > threshold_val:
                reason = "MV_OUTLIER_OD"
                score_val = od_i

        rows.append(
            {
                "sample_id": sample_ids[i],
                "method": policy.threshold_strategy,
                "score": score_val if score_val is not None else (sd_i if sd_i is not None else od_i),
                "threshold": threshold_val,
                "flag": reason is not None,
                "reason_code": reason or "",
                "action": policy.action,
                "severity": policy.severity,
            }
        )

    return pd.DataFrame(rows)


def batch_drift(
    Z: Any,
    batch_labels: Iterable[Any],
    metric: Literal["centroid_l2", "mmd_rbf", "wasserstein_approx"] = "centroid_l2",
    warn_threshold: float = 2.0,
    fail_threshold: float = 4.0,
) -> pd.DataFrame:
    """Compute batch drift metrics (centroid distance supported)."""

    arr = _coerce_scores(Z)
    labels = np.asarray(list(batch_labels))
    if len(labels) != arr.shape[0]:
        raise ValueError("batch_labels length must match rows in Z")
    global_centroid = np.mean(arr, axis=0)
    rows = []
    for batch in np.unique(labels):
        mask = labels == batch
        if not np.any(mask):
            continue
        centroid = np.mean(arr[mask], axis=0)
        value = float(np.linalg.norm(centroid - global_centroid))
        status = "ok"
        if value >= fail_threshold:
            status = "fail"
        elif value >= warn_threshold:
            status = "warn"
        rows.append({"group": batch, "metric": metric, "value": value, "status": status})
    return pd.DataFrame(rows)


def score_outliers(scores: Any, sample_ids: Sequence[Any], threshold: float = 3.0) -> Tuple[pd.DataFrame, dict]:
    """Backwards-compatible z-score outlier detector."""

    arr = _coerce_scores(scores)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=1)
    std_safe = np.where(std == 0, 1e-9, std)
    z = np.abs((arr - mean) / std_safe)
    max_z = np.max(z, axis=1)
    flags = max_z > threshold
    table = pd.DataFrame({"sample_id": list(sample_ids), "max_zscore": max_z, "is_outlier": flags})
    summary = {"outliers": int(flags.sum()), "threshold": threshold}
    return table, summary


def centroid_shift(scores: Any, groups: Optional[Iterable[Any]]) -> Optional[pd.DataFrame]:
    """Backwards-compatible centroid distance summary."""

    if groups is None:
        return None
    drift_df = batch_drift(scores, groups, warn_threshold=0.0, fail_threshold=float("inf"))
    return drift_df if not drift_df.empty else None


__all__ = [
    "compute_pca_outlier_scores",
    "hotelling_t2",
    "outlier_flags",
    "batch_drift",
    "MultivariateQCPolicy",
    "summarize_scores",
    "score_outliers",
    "centroid_shift",
]
