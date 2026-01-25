"""
Regulatory readiness and auditability helpers.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def validate_reproducibility(manifest: Mapping[str, object]) -> None:
    """
    Validate that run metadata is sufficient for reproducibility/audit.

    Required fields (actionable errors if missing):
    - protocol_hash (str)
    - seed (int)
    - dependencies (mapping of package -> version)
    - data_fingerprint (str) uniquely identifying input data snapshot
    """

    if manifest is None:
        raise ValueError("manifest is required for audit trail validation")

    if not manifest.get("protocol_hash"):
        raise ValueError("protocol_hash missing in manifest; cannot prove protocol version")

    if "seed" not in manifest:
        raise ValueError("seed missing in manifest; set a deterministic seed and record it")

    deps = manifest.get("dependencies")
    if not deps or not isinstance(deps, Mapping) or len(deps) == 0:
        raise ValueError("dependency versions missing; record package versions in manifest['dependencies']")

    if not manifest.get("data_fingerprint"):
        raise ValueError("data_fingerprint missing; store an input data hash or checksum")


def integrity_checks(
    predictions_df: pd.DataFrame,
    prob_prefix: str = "prob_",
    tol: float = 1e-6,
) -> None:
    """
    Validate prediction outputs for auditability.

    Checks
    ------
    - No NaNs anywhere
    - Probability columns (prefixed by prob_prefix) exist and rows sum to 1 within tolerance
    - If ``predicted_class`` present, it matches argmax of probabilities
    - If ``predicted_label`` present and probability columns encode labels (prob_<label>), ensure consistency
    """

    if predictions_df.isna().any().any():
        raise ValueError("predictions contain NaN values; clean or regenerate outputs")

    prob_cols = [c for c in predictions_df.columns if c.startswith(prob_prefix)]
    if not prob_cols:
        raise ValueError(f"no probability columns found (expected prefix '{prob_prefix}')")

    probs = predictions_df[prob_cols].to_numpy(dtype=float)
    row_sums = probs.sum(axis=1)
    bad_sum_idx = np.where(np.abs(row_sums - 1.0) > tol)[0]
    if bad_sum_idx.size > 0:
        first_idx = bad_sum_idx[0]
        raise ValueError(
            f"probabilities must sum to 1 (row {first_idx} sums to {row_sums[first_idx]:.6f})"
        )

    # Map probability columns to labels if possible (prob_<label>)
    label_from_col: Dict[int, str] = {}
    label_names: Optional[list[str]] = None
    try:
        label_names = [c[len(prob_prefix):] for c in prob_cols]
    except Exception:
        label_names = None

    # predicted_class consistency
    if "predicted_class" in predictions_df.columns:
        predicted_class = predictions_df["predicted_class"].to_numpy()
        argmax_indices = probs.argmax(axis=1)
        mismatch = np.where(predicted_class != argmax_indices)[0]
        if mismatch.size > 0:
            idx = mismatch[0]
            raise ValueError(
                f"predicted_class mismatch at row {idx}: argmax={argmax_indices[idx]}, predicted={predicted_class[idx]}"
            )

    # predicted_label consistency (only if we can map labels)
    if label_names is not None and "predicted_label" in predictions_df.columns:
        predicted_label = predictions_df["predicted_label"].astype(str).to_numpy()
        argmax_labels = np.array([label_names[i] for i in probs.argmax(axis=1)])
        mismatch = np.where(predicted_label != argmax_labels)[0]
        if mismatch.size > 0:
            idx = mismatch[0]
            raise ValueError(
                f"predicted_label mismatch at row {idx}: argmax='{argmax_labels[idx]}', predicted='{predicted_label[idx]}'"
            )


def generate_trust_summary(
    metrics_summary: Mapping[str, object],
    coverage_table: Optional[pd.DataFrame],
    calibration_metrics: Mapping[str, object],
    abstention_metrics: Mapping[str, object],
) -> Dict[str, object]:
    """
    Build a concise trust/audit summary for embedding in reports.
    """

    coverage_records: Iterable[Dict[str, object]]
    if coverage_table is None:
        coverage_records = []
    else:
        coverage_records = coverage_table.to_dict(orient="records")

    return {
        "metrics_summary": dict(metrics_summary),
        "coverage_table": list(coverage_records),
        "calibration_metrics": dict(calibration_metrics),
        "abstention_metrics": dict(abstention_metrics),
    }


__all__ = [
    "validate_reproducibility",
    "integrity_checks",
    "generate_trust_summary",
]
