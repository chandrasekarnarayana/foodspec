"""
Class balance diagnostics for preventing silent bias in food spectroscopy ML.

Most food science datasets suffer from extreme class imbalance due to unequal
sampling costs (e.g., 100 authentic oils vs 5 adulterants). Silent imbalance
leads to high-accuracy classifiers that always predict the majority class.

**Key Assumptions:**
1. Classes are discrete and mutually exclusive
2. Label column exists in metadata
3. Minimum recommended samples per class: 20 (for robust modeling)
4. Severe imbalance threshold: >10:1 ratio
5. Stratified splitting required for imbalanced data

**Typical Usage:**
    >>> from foodspec import FoodSpec
    >>> fs = FoodSpec("oils.csv", modality="raman")
    >>> balance = fs.check_class_balance(label_column="oil_type")
    >>> if balance["severe_imbalance"]:
    >>>     print(f"⚠️ Imbalance ratio: {balance['imbalance_ratio']:.1f}:1")
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet


def check_class_balance(
    metadata: pd.DataFrame,
    label_column: str,
    severe_threshold: float = 10.0,
    min_samples_per_class: int = 20,
) -> Dict[str, Any]:
    """
    Check class balance and flag severe imbalance.

    **Assumptions:**
    - label_column is categorical
    - severe_threshold is ratio (max_class / min_class)
    - min_samples_per_class is minimum viable for ML training

    Parameters
    ----------
    metadata : pd.DataFrame
        Dataset metadata.
    label_column : str
        Column with class labels.
    severe_threshold : float, default=10.0
        Imbalance ratio above which to flag as severe.
    min_samples_per_class : int, default=20
        Minimum recommended samples per class.

    Returns
    -------
    metrics : dict
        - 'samples_per_class': dict mapping class -> count
        - 'imbalance_ratio': max/min class size
        - 'severe_imbalance': bool, True if ratio > threshold
        - 'undersized_classes': list of classes with < min_samples_per_class
        - 'majority_class': (class_name, count)
        - 'minority_class': (class_name, count)
        - 'recommended_action': string recommendation
    """
    if label_column not in metadata.columns:
        raise ValueError(f"label_column '{label_column}' not found in metadata.")

    labels = metadata[label_column]
    class_counts = labels.value_counts(dropna=True).to_dict()

    if len(class_counts) == 0:
        raise ValueError(f"No valid labels found in '{label_column}'.")

    counts_array = np.array(list(class_counts.values()))
    min_count = int(counts_array.min())
    max_count = int(counts_array.max())
    imbalance_ratio = max_count / (min_count + 1e-12)

    # Identify majority and minority
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # Undersized classes
    undersized = [cls for cls, count in class_counts.items() if count < min_samples_per_class]

    # Severe imbalance flag
    severe = imbalance_ratio > severe_threshold

    # Recommendation
    if severe and undersized:
        recommendation = (
            f"⚠️ Severe imbalance ({imbalance_ratio:.1f}:1) with undersized classes. "
            "Consider: (1) stratified sampling, (2) oversampling minority classes, "
            "(3) class_weight='balanced' in sklearn models."
        )
    elif severe:
        recommendation = (
            f"⚠️ Severe imbalance ({imbalance_ratio:.1f}:1). Use stratified splitting and class_weight='balanced'."
        )
    elif undersized:
        recommendation = (
            f"Classes {undersized} have < {min_samples_per_class} samples. "
            "Consider collecting more data or using data augmentation."
        )
    else:
        recommendation = "✓ Class balance is adequate for standard ML workflows."

    if severe or undersized:
        warnings.warn(recommendation)

    metrics = {
        "samples_per_class": {str(k): int(v) for k, v in class_counts.items()},
        "imbalance_ratio": float(imbalance_ratio),
        "severe_imbalance": bool(severe),
        "undersized_classes": [str(c) for c in undersized],
        "majority_class": (str(majority_class), int(class_counts[majority_class])),
        "minority_class": (str(minority_class), int(class_counts[minority_class])),
        "recommended_action": recommendation,
    }

    return metrics


def diagnose_imbalance(
    dataset: FoodSpectrumSet,
    label_column: str,
    stratification_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Detailed class imbalance diagnostics with optional stratification analysis.

    **Workflow:**
    1. Overall class balance check
    2. If stratification_column provided: check balance within each stratum (e.g., batch)
    3. Flag problematic batches with severe within-batch imbalance

    **Assumptions:**
    - stratification_column (if provided) defines independent sampling units (batches)
    - Within-batch imbalance suggests systematic collection bias

    Parameters
    ----------
    dataset : FoodSpectrumSet
        Input dataset.
    label_column : str
        Column with class labels.
    stratification_column : str, optional
        Column defining strata (e.g., 'batch_id', 'collection_date').

    Returns
    -------
    diagnostics : dict
        - 'overall_balance': metrics from check_class_balance()
        - 'stratified_balance' (if stratification_column): per-stratum balance
        - 'problematic_strata': list of strata with severe imbalance
    """
    metadata = dataset.metadata

    # Overall balance
    overall_balance = check_class_balance(metadata, label_column)

    diagnostics = {
        "overall_balance": overall_balance,
    }

    # Stratified analysis
    if stratification_column is not None:
        if stratification_column not in metadata.columns:
            raise ValueError(f"stratification_column '{stratification_column}' not found.")

        strata_values = metadata[stratification_column].unique()
        stratified_balance = {}
        problematic_strata = []

        for stratum in strata_values:
            stratum_mask = metadata[stratification_column] == stratum
            stratum_meta = metadata[stratum_mask]

            if len(stratum_meta) < 2:
                continue  # Skip tiny strata

            try:
                stratum_balance = check_class_balance(stratum_meta, label_column)
                stratified_balance[str(stratum)] = stratum_balance

                if stratum_balance["severe_imbalance"]:
                    problematic_strata.append(str(stratum))
            except ValueError:
                # Stratum may have only one class
                continue

        diagnostics["stratified_balance"] = stratified_balance
        diagnostics["problematic_strata"] = problematic_strata

        if problematic_strata:
            warnings.warn(
                f"Severe imbalance detected in {len(problematic_strata)} strata: {problematic_strata}. "
                "This suggests systematic collection bias."
            )

    return diagnostics
