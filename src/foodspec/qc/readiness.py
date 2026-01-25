from __future__ import annotations
"""
Dataset readiness score for ML deployment gatekeeping.

Provides a single 0-100 score summarizing dataset quality across multiple
dimensions. Used in industry to gate model training: score < 70 → fix data first.

**Scoring Dimensions:**
1. Sample size (20% weight): sufficient samples per class
2. Class balance (20% weight): imbalance ratio < 10:1
3. Replicate consistency (15% weight): technical CV < 10%
4. Metadata completeness (15% weight): required fields non-null
5. Spectral quality (15% weight): SNR, no NaN/inf
6. Leakage risk (15% weight): no batch–label confounding

**Key Assumptions:**
1. All dimensions equally important (can be weighted differently)
2. Thresholds based on FoodSpec community best practices
3. Score ≥80: production-ready
4. Score 60-80: usable with caveats
5. Score <60: high risk, data improvement needed

**Typical Usage:**
    >>> from foodspec import FoodSpec
    >>> fs = FoodSpec("data.csv", modality="raman")
    >>> readiness = fs.compute_readiness_score(
    ...     label_column="oil_type",
    ...     batch_column="collection_batch",
    ...     replicate_column="sample_id"
    ... )
    >>> print(f"Readiness: {readiness['overall_score']:.0f}/100")
    >>> if readiness['overall_score'] < 70:
    >>>     print(f"Blockers: {readiness['failed_criteria']}")
"""


import warnings
from typing import Any, Dict, Optional

import numpy as np

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.core.summary import (
    compute_metadata_completeness,
    compute_samples_per_class,
    compute_spectral_quality_metrics,
)
from foodspec.qc.dataset_qc import check_class_balance
from foodspec.qc.leakage import detect_batch_label_correlation
from foodspec.qc.replicates import compute_replicate_consistency


def _score_sample_size(
    n_samples: int,
    min_per_class: int,
    n_classes: int,
) -> float:
    """
    Score sample size: 100 if all classes have ≥ min_per_class, else scaled.

    Assumes min_per_class is the target (typically 20-30 for ML).
    """
    if min_per_class >= (n_samples / n_classes):
        # Insufficient samples
        actual_per_class = n_samples / n_classes
        score = (actual_per_class / min_per_class) * 100
    else:
        score = 100.0

    return min(score, 100.0)


def _score_class_balance(imbalance_ratio: float, threshold: float = 10.0) -> float:
    """
    Score class balance: 100 if ratio ≤ threshold, else penalized.

    Assumes threshold is max acceptable ratio (e.g., 10:1).
    """
    if imbalance_ratio <= threshold:
        score = 100.0
    else:
        # Logarithmic penalty for severe imbalance
        score = 100.0 - (np.log10(imbalance_ratio / threshold) * 40)

    return max(score, 0.0)


def _score_replicate_consistency(median_cv: float, threshold: float = 10.0) -> float:
    """
    Score replicate consistency: 100 if CV ≤ threshold, else penalized.

    Assumes threshold is max acceptable technical CV (%).
    """
    if np.isnan(median_cv):
        # No replicates: assume perfect (or skip this dimension)
        return 100.0

    if median_cv <= threshold:
        score = 100.0
    else:
        # Linear penalty
        score = 100.0 - ((median_cv - threshold) * 3)

    return max(score, 0.0)


def _score_metadata_completeness(completeness: float) -> float:
    """
    Score metadata completeness: 100 if all fields complete, else scaled.

    completeness ∈ [0, 1]: fraction of non-null cells.
    """
    score = completeness * 100
    return score


def _score_spectral_quality(
    mean_snr: float,
    nan_count: int,
    inf_count: int,
    negative_rate: float,
) -> float:
    """
    Score spectral quality based on SNR and data validity.

    Assumes:
    - SNR > 10: good
    - SNR 5-10: acceptable
    - SNR < 5: poor
    - No NaN/inf: +20 points
    - Negative rate < 1%: +10 points
    """
    # SNR component (0-70 points)
    if mean_snr > 10:
        snr_score = 70.0
    elif mean_snr > 5:
        snr_score = 35.0 + ((mean_snr - 5) / 5) * 35
    else:
        snr_score = (mean_snr / 5) * 35

    # Validity component (0-20 points)
    if nan_count == 0 and inf_count == 0:
        validity_score = 20.0
    else:
        validity_score = 0.0

    # Negative intensity component (0-10 points)
    if negative_rate < 0.01:
        negative_score = 10.0
    elif negative_rate < 0.05:
        negative_score = 5.0
    else:
        negative_score = 0.0

    score = snr_score + validity_score + negative_score
    return min(score, 100.0)


def _score_leakage_risk(
    cramers_v: Optional[float],
    replicate_leakage_risk: Optional[str],
) -> float:
    """
    Score leakage risk: 100 if no leakage, else penalized.

    Assumes:
    - Cramér's V < 0.3: minimal confounding
    - Cramér's V 0.3-0.5: moderate
    - Cramér's V > 0.5: severe
    - Replicate leakage risk "high": -30 points
    """
    score = 100.0

    # Batch–label correlation penalty
    if cramers_v is not None:
        if cramers_v > 0.5:
            score -= 50.0
        elif cramers_v > 0.3:
            score -= 25.0

    # Replicate leakage penalty
    if replicate_leakage_risk == "high":
        score -= 30.0

    return max(score, 0.0)


def compute_readiness_score(
    dataset: FoodSpectrumSet,
    label_column: str,
    batch_column: Optional[str] = None,
    replicate_column: Optional[str] = None,
    required_metadata_columns: Optional[list] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive dataset readiness score (0-100).

    **Workflow:**
    1. Score sample size
    2. Score class balance
    3. Score replicate consistency (if replicate_column provided)
    4. Score metadata completeness
    5. Score spectral quality
    6. Score leakage risk (if batch/replicate columns provided)
    7. Weighted average → overall score

    **Assumptions:**
    - Default weights: sample_size=0.20, balance=0.20, replicates=0.15,
      metadata=0.15, spectral=0.15, leakage=0.15
    - Thresholds: min_samples_per_class=20, imbalance_ratio=10, technical_cv=10%

    Parameters
    ----------
    dataset : FoodSpectrumSet
        Input dataset.
    label_column : str
        Column with class labels.
    batch_column : str, optional
        Column defining batches (for leakage detection).
    replicate_column : str, optional
        Column defining replicate groups (for consistency and leakage).
    required_metadata_columns : list of str, optional
        Metadata columns that must be complete.
    weights : dict, optional
        Custom weights for scoring dimensions.
        Keys: 'sample_size', 'balance', 'replicates', 'metadata', 'spectral', 'leakage'

    Returns
    -------
    score_report : dict
        - 'overall_score': 0-100 score
        - 'dimension_scores': dict with individual dimension scores
        - 'passed_criteria': list of criteria that passed (score ≥ 70)
        - 'failed_criteria': list of criteria that failed (score < 70)
        - 'recommendation': text guidance based on overall score
    """
    metadata = dataset.metadata
    spectra = dataset.x
    wavenumbers = dataset.wavenumbers

    # Default weights
    default_weights = {
        "sample_size": 0.20,
        "balance": 0.20,
        "replicates": 0.15,
        "metadata": 0.15,
        "spectral": 0.15,
        "leakage": 0.15,
    }
    weights = weights or default_weights

    dimension_scores = {}

    # 1. Sample size
    class_info = compute_samples_per_class(metadata, label_column)
    sample_size_score = _score_sample_size(
        class_info["total_samples"],
        class_info["min_class_size"],
        class_info["n_classes"],
    )
    dimension_scores["sample_size"] = sample_size_score

    # 2. Class balance
    balance_info = check_class_balance(metadata, label_column)
    balance_score = _score_class_balance(balance_info["imbalance_ratio"])
    dimension_scores["balance"] = balance_score

    # 3. Replicate consistency
    if replicate_column is not None:
        rep_consistency = compute_replicate_consistency(spectra, metadata, replicate_column)
        replicates_score = _score_replicate_consistency(rep_consistency["median_cv"])
    else:
        replicates_score = 100.0  # Assume OK if not provided
        warnings.warn("No replicate_column provided; skipping replicate consistency scoring.")
    dimension_scores["replicates"] = replicates_score

    # 4. Metadata completeness
    meta_completeness = compute_metadata_completeness(metadata, required_metadata_columns)
    metadata_score = _score_metadata_completeness(meta_completeness["overall_completeness"])
    dimension_scores["metadata"] = metadata_score

    # 5. Spectral quality
    spectral_quality = compute_spectral_quality_metrics(spectra, wavenumbers, dataset.modality)
    spectral_score = _score_spectral_quality(
        spectral_quality["mean_snr"],
        spectral_quality["nan_count"],
        spectral_quality["inf_count"],
        spectral_quality["negative_intensity_rate"],
    )
    dimension_scores["spectral"] = spectral_score

    # 6. Leakage risk
    cramers_v = None
    rep_leak_risk = None

    if batch_column is not None:
        batch_corr = detect_batch_label_correlation(metadata, label_column, batch_column)
        cramers_v = batch_corr["cramers_v"]

    if replicate_column is not None:
        # For leakage risk, just check replicate group sizes
        replicate_ids = metadata[replicate_column]
        replicate_sizes = replicate_ids.value_counts()
        multi_sample_groups = (replicate_sizes > 1).sum()
        rep_leak_risk = "high" if multi_sample_groups > 0 else "low"

    leakage_score = _score_leakage_risk(cramers_v, rep_leak_risk)
    dimension_scores["leakage"] = leakage_score

    # Overall weighted score
    overall_score = sum(dimension_scores[dim] * weights.get(dim, 0.0) for dim in dimension_scores)

    # Passed/failed criteria (threshold = 70)
    passed = [dim for dim, score in dimension_scores.items() if score >= 70]
    failed = [dim for dim, score in dimension_scores.items() if score < 70]

    # Recommendation
    if overall_score >= 80:
        recommendation = "✓ Dataset is production-ready for ML deployment. Proceed with model training and validation."
    elif overall_score >= 60:
        recommendation = (
            f"⚠️ Dataset is usable with caveats. Failed criteria: {failed}. "
            "Consider addressing these issues before production deployment."
        )
    else:
        recommendation = (
            f"❌ Dataset is high-risk for ML. Failed criteria: {failed}. "
            "Data improvement required before model training."
        )

    score_report = {
        "overall_score": float(overall_score),
        "dimension_scores": {k: float(v) for k, v in dimension_scores.items()},
        "passed_criteria": passed,
        "failed_criteria": failed,
        "recommendation": recommendation,
    }

    return score_report
