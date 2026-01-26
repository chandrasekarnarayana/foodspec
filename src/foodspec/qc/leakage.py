from __future__ import annotations
"""
Data leakage detection for food spectroscopy ML workflows.

Leakage occurs when test-set information "leaks" into training, causing
overly optimistic performance estimates. Common sources:
1. Replicates split across train/test (identical samples in both sets)
2. Batch effects confounded with class labels (model learns batch, not biology)
3. Temporal correlation (time-series data split randomly)

**Key Assumptions:**
1. Leakage is dataset-level, not model-level (detected before training)
2. Batch–label correlation measured via chi-squared or Cramér's V
3. Replicate leakage requires explicit replicate_column in metadata
4. Threshold for "severe" correlation: Cramér's V > 0.5

**Typical Usage:**
    >>> from foodspec import FoodSpec
    >>> fs = FoodSpec("data.csv", modality="raman")
    >>> leakage = fs.detect_leakage(
    ...     label_column="oil_type",
    ...     batch_column="collection_batch",
    ...     replicate_column="sample_id"
    ... )
    >>> if leakage["batch_label_correlation"]["severe"]:
    >>>     print("⚠️ Batch effects confounded with labels!")
"""


import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from foodspec.data_objects.spectra_set import FoodSpectrumSet


def detect_batch_label_correlation(
    metadata: pd.DataFrame,
    label_column: str,
    batch_column: str,
    severe_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect correlation between batch and class labels (confounding).

    **Assumptions:**
    - batch_column and label_column are categorical
    - Chi-squared test assumptions apply (expected counts ≥5)
    - Cramér's V ∈ [0, 1]: 0=no association, 1=perfect association

    Parameters
    ----------
    metadata : pd.DataFrame
        Dataset metadata.
    label_column : str
        Column with class labels.
    batch_column : str
        Column defining batches.
    severe_threshold : float, default=0.5
        Cramér's V above which to flag as severe confounding.

    Returns
    -------
    metrics : dict
        - 'cramers_v': Cramér's V statistic
        - 'chi2_pvalue': p-value from chi-squared test
        - 'severe': bool, True if Cramér's V > threshold
        - 'contingency_table': batch × label counts
        - 'interpretation': text explanation
    """
    if label_column not in metadata.columns:
        raise ValueError(f"label_column '{label_column}' not found.")
    if batch_column not in metadata.columns:
        raise ValueError(f"batch_column '{batch_column}' not found.")

    # Contingency table
    contingency = pd.crosstab(metadata[batch_column], metadata[label_column])

    # Chi-squared test
    chi2, pvalue, dof, expected = chi2_contingency(contingency)

    # Cramér's V: sqrt(chi2 / (n * min(r-1, c-1)))
    n = contingency.sum().sum()
    r, c = contingency.shape
    cramers_v = np.sqrt(chi2 / (n * min(r - 1, c - 1)))

    severe = cramers_v > severe_threshold

    # Interpretation
    if severe:
        interpretation = (
            f"⚠️ Severe batch–label correlation (Cramér's V = {cramers_v:.3f}). "
            "Batches are confounded with classes. Models may learn batch artifacts "
            "instead of true biological signals. Consider batch correction or "
            "stratified sampling."
        )
        warnings.warn(interpretation)
    elif cramers_v > 0.3:
        interpretation = (
            f"Moderate batch–label correlation (Cramér's V = {cramers_v:.3f}). "
            "Some confounding present; use batch-aware cross-validation."
        )
    else:
        interpretation = (
            f"Minimal batch–label correlation (Cramér's V = {cramers_v:.3f}). "
            "Batches are approximately balanced across classes."
        )

    metrics = {
        "cramers_v": float(cramers_v),
        "chi2_pvalue": float(pvalue),
        "severe": bool(severe),
        "contingency_table": contingency.to_dict(),
        "interpretation": interpretation,
    }

    return metrics


def detect_replicate_leakage(
    metadata: pd.DataFrame,
    replicate_column: str,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Detect replicate leakage in train/test splits.

    **Assumptions:**
    - replicate_column defines groups that should NOT be split
    - train_indices and test_indices (if provided) are row indices into metadata
    - If indices not provided: check potential for leakage (replicate group sizes)

    Parameters
    ----------
    metadata : pd.DataFrame
        Dataset metadata.
    replicate_column : str
        Column defining replicate groups.
    train_indices : np.ndarray, optional
        Row indices for training set.
    test_indices : np.ndarray, optional
        Row indices for test set.

    Returns
    -------
    metrics : dict
        - 'replicate_groups': total number of replicate groups
        - 'mean_group_size': average replicate group size
        - 'leakage_risk': 'high' if any group > 1 sample (without explicit split)
        - 'leaked_groups' (if train/test provided): groups split across train/test
        - 'n_leaked_groups': count of leaked groups
        - 'recommended_action': text guidance
    """
    if replicate_column not in metadata.columns:
        raise ValueError(f"replicate_column '{replicate_column}' not found.")

    replicate_ids = metadata[replicate_column]
    unique_replicates = replicate_ids.unique()
    replicate_sizes = replicate_ids.value_counts().to_dict()

    mean_group_size = np.mean(list(replicate_sizes.values()))

    # High risk: any replicate group > 1 (potential for splitting)
    multi_sample_groups = [rep for rep, size in replicate_sizes.items() if size > 1]
    leakage_risk = "high" if multi_sample_groups else "low"

    metrics = {
        "replicate_groups": len(unique_replicates),
        "mean_group_size": float(mean_group_size),
        "leakage_risk": leakage_risk,
    }

    # If train/test indices provided: check actual leakage
    if train_indices is not None and test_indices is not None:
        train_reps = set(replicate_ids.iloc[train_indices])
        test_reps = set(replicate_ids.iloc[test_indices])

        leaked_groups = train_reps.intersection(test_reps)
        n_leaked = len(leaked_groups)

        metrics["leaked_groups"] = list(leaked_groups)
        metrics["n_leaked_groups"] = n_leaked

        if n_leaked > 0:
            recommendation = (
                f"⚠️ {n_leaked} replicate groups split across train/test sets. "
                "Use GroupKFold or GroupShuffleSplit from sklearn to prevent leakage."
            )
            warnings.warn(recommendation)
        else:
            recommendation = "✓ No replicate leakage detected in train/test split."

        metrics["recommended_action"] = recommendation
    else:
        # No split provided: warn about potential
        if leakage_risk == "high":
            recommendation = (
                f"{len(multi_sample_groups)} replicate groups have multiple samples. "
                "Use GroupKFold or GroupShuffleSplit to prevent leakage when splitting."
            )
        else:
            recommendation = "✓ Low leakage risk (all replicates are singleton groups)."

        metrics["recommended_action"] = recommendation

    return metrics


def detect_leakage(
    dataset: FoodSpectrumSet,
    label_column: str,
    batch_column: Optional[str] = None,
    replicate_column: Optional[str] = None,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Comprehensive leakage detection for dataset governance.

    **Workflow:**
    1. Batch–label correlation (if batch_column provided)
    2. Replicate leakage risk (if replicate_column provided)
    3. Actual leakage check (if train/test indices provided)

    Parameters
    ----------
    dataset : FoodSpectrumSet
        Input dataset.
    label_column : str
        Column with class labels.
    batch_column : str, optional
        Column defining batches.
    replicate_column : str, optional
        Column defining replicate groups.
    train_indices : np.ndarray, optional
        Row indices for training set.
    test_indices : np.ndarray, optional
        Row indices for test set.

    Returns
    -------
    leakage_report : dict
        - 'batch_label_correlation' (if batch_column): correlation metrics
        - 'replicate_leakage' (if replicate_column): leakage risk/detection
        - 'overall_risk': 'high', 'moderate', 'low'
    """
    metadata = dataset.metadata
    leakage_report = {}

    risk_flags = []

    # Batch–label correlation
    if batch_column is not None:
        batch_corr = detect_batch_label_correlation(metadata, label_column, batch_column)
        leakage_report["batch_label_correlation"] = batch_corr

        if batch_corr["severe"]:
            risk_flags.append("severe_batch_confounding")

    # Replicate leakage
    if replicate_column is not None:
        rep_leak = detect_replicate_leakage(metadata, replicate_column, train_indices, test_indices)
        leakage_report["replicate_leakage"] = rep_leak

        if rep_leak["leakage_risk"] == "high":
            risk_flags.append("replicate_leakage_risk")

        if "n_leaked_groups" in rep_leak and rep_leak["n_leaked_groups"] > 0:
            risk_flags.append("actual_replicate_leakage")

    # Overall risk
    if "actual_replicate_leakage" in risk_flags or "severe_batch_confounding" in risk_flags:
        overall_risk = "high"
    elif risk_flags:
        overall_risk = "moderate"
    else:
        overall_risk = "low"

    leakage_report["overall_risk"] = overall_risk

    return leakage_report
