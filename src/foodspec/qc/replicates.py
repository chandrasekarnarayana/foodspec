"""
Replicate consistency analysis for assessing experimental reliability.

Technical replicates (same sample, multiple measurements) vs biological replicates
(different samples, same conditions) have different variability expectations.
High technical replicate CV suggests instrument instability; high biological
CV is expected but must be quantified for ML uncertainty estimation.

**Key Assumptions:**
1. Replicate groups are defined by a metadata column (e.g., 'sample_id')
2. Technical replicates have identical sample_id; biologicals have unique IDs
3. Expected technical CV: <5% for FTIR, <10% for Raman
4. Expected biological CV: 10-30% depending on food matrix heterogeneity
5. Replicate groups have ≥2 measurements

**Typical Usage:**
    >>> from foodspec import FoodSpec
    >>> fs = FoodSpec("samples.csv", modality="raman")
    >>> consistency = fs.assess_replicate_consistency(replicate_column="sample_id")
    >>> print(f"Median technical CV: {consistency['median_cv']:.1f}%")
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from foodspec.core.dataset import FoodSpectrumSet


def compute_replicate_consistency(
    spectra: np.ndarray,
    metadata: pd.DataFrame,
    replicate_column: str,
    technical_cv_threshold: float = 10.0,
) -> Dict[str, Any]:
    """
    Compute coefficient of variation (CV) for replicate groups.

    **Assumptions:**
    - replicate_column defines groups of repeated measurements
    - Within-group variability is primarily technical (for true replicates)
    - CV = (std / mean) * 100 computed per wavenumber, then averaged

    Parameters
    ----------
    spectra : np.ndarray, shape (n_samples, n_wavenumbers)
        Spectral data.
    metadata : pd.DataFrame
        Metadata with replicate identifiers.
    replicate_column : str
        Column defining replicate groups.
    technical_cv_threshold : float, default=10.0
        CV (%) above which to flag as high variability.

    Returns
    -------
    metrics : dict
        - 'cv_per_replicate': dict mapping replicate_id -> CV (%)
        - 'median_cv': median CV across all replicates
        - 'mean_cv': mean CV across all replicates
        - 'max_cv': maximum CV (worst replicate)
        - 'high_variability_replicates': list of replicate IDs with CV > threshold
        - 'n_replicates': number of replicate groups evaluated
    """
    if replicate_column not in metadata.columns:
        raise ValueError(f"replicate_column '{replicate_column}' not found in metadata.")

    replicate_ids = metadata[replicate_column]
    unique_replicates = replicate_ids.unique()

    cv_dict = {}
    high_variability = []

    for rep_id in unique_replicates:
        rep_mask = replicate_ids == rep_id
        rep_spectra = spectra[rep_mask]

        if rep_spectra.shape[0] < 2:
            continue  # Need at least 2 measurements

        # CV per wavenumber, then average
        means = rep_spectra.mean(axis=0)
        stds = rep_spectra.std(axis=0, ddof=1)

        # Avoid division by zero
        valid_mask = means > 1e-12
        cv_per_wn = np.where(valid_mask, (stds / means) * 100, 0)

        mean_cv = cv_per_wn.mean()
        cv_dict[str(rep_id)] = float(mean_cv)

        if mean_cv > technical_cv_threshold:
            high_variability.append(str(rep_id))

    if len(cv_dict) == 0:
        warnings.warn(f"No replicate groups with ≥2 measurements found in '{replicate_column}'.")
        return {
            "cv_per_replicate": {},
            "median_cv": float("nan"),
            "mean_cv": float("nan"),
            "max_cv": float("nan"),
            "high_variability_replicates": [],
            "n_replicates": 0,
        }

    cv_values = np.array(list(cv_dict.values()))
    median_cv = float(np.median(cv_values))
    mean_cv = float(np.mean(cv_values))
    max_cv = float(np.max(cv_values))

    if high_variability:
        warnings.warn(
            f"{len(high_variability)} replicate groups have CV > {technical_cv_threshold}%: {high_variability[:5]}... "
            "This suggests instrument instability or sample handling variability."
        )

    metrics = {
        "cv_per_replicate": cv_dict,
        "median_cv": median_cv,
        "mean_cv": mean_cv,
        "max_cv": max_cv,
        "high_variability_replicates": high_variability,
        "n_replicates": len(cv_dict),
    }

    return metrics


def assess_variability_sources(
    dataset: FoodSpectrumSet,
    replicate_column: str,
    label_column: Optional[str] = None,
    batch_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decompose variability into technical, biological, and batch components.

    **Workflow:**
    1. Compute replicate CV (technical variability)
    2. If label_column: compute within-class variability (biological)
    3. If batch_column: compute between-batch variability
    4. Report variance partitioning: technical vs biological vs batch

    **Assumptions:**
    - Technical variance < biological variance < batch variance (ideally)
    - ANOVA assumptions apply (normality, homoscedasticity)

    Parameters
    ----------
    dataset : FoodSpectrumSet
        Input dataset.
    replicate_column : str
        Column defining replicate groups.
    label_column : str, optional
        Column with class labels (for biological variability).
    batch_column : str, optional
        Column defining batches (for batch effects).

    Returns
    -------
    diagnostics : dict
        - 'technical_variability': replicate consistency metrics
        - 'biological_variability' (if label_column): within-class std
        - 'batch_variability' (if batch_column): between-batch variance ratio
        - 'variance_partition': relative contribution of each source (%)
    """
    metadata = dataset.metadata
    spectra = dataset.x

    diagnostics = {}

    # Technical variability
    tech_var = compute_replicate_consistency(spectra, metadata, replicate_column)
    diagnostics["technical_variability"] = tech_var

    # Biological variability (within-class)
    if label_column is not None:
        if label_column not in metadata.columns:
            raise ValueError(f"label_column '{label_column}' not found.")

        labels = metadata[label_column]
        unique_classes = labels.unique()

        within_class_std = []
        for cls in unique_classes:
            cls_mask = labels == cls
            cls_spectra = spectra[cls_mask]

            if cls_spectra.shape[0] < 2:
                continue

            std_per_wn = cls_spectra.std(axis=0, ddof=1)
            within_class_std.append(std_per_wn.mean())

        if within_class_std:
            bio_var = {
                "mean_within_class_std": float(np.mean(within_class_std)),
                "median_within_class_std": float(np.median(within_class_std)),
            }
            diagnostics["biological_variability"] = bio_var

    # Batch variability
    if batch_column is not None:
        if batch_column not in metadata.columns:
            raise ValueError(f"batch_column '{batch_column}' not found.")

        batches = metadata[batch_column]
        unique_batches = batches.unique()

        if len(unique_batches) < 2:
            warnings.warn("Need ≥2 batches for batch variability analysis.")
        else:
            # Between-batch variance ratio (F-statistic from one-way ANOVA)
            # Average across wavenumbers
            f_stats = []
            for wn_idx in range(spectra.shape[1]):
                intensities = spectra[:, wn_idx]
                batch_groups = [intensities[batches == b] for b in unique_batches if (batches == b).sum() > 0]

                if len(batch_groups) < 2:
                    continue

                try:
                    f_stat, _ = stats.f_oneway(*batch_groups)
                    f_stats.append(f_stat)
                except Exception:  # noqa: BLE001
                    continue

            if f_stats:
                batch_var = {
                    "mean_f_statistic": float(np.mean(f_stats)),
                    "median_f_statistic": float(np.median(f_stats)),
                    "interpretation": ("F > 2: significant batch effects. F < 1.5: batch effects negligible."),
                }
                diagnostics["batch_variability"] = batch_var

    # Variance partitioning (rough estimate)
    # Tech CV → technical variance
    # Bio std → biological variance
    # Batch F-stat → batch variance
    tech_cv = tech_var["median_cv"]
    bio_std = diagnostics.get("biological_variability", {}).get("mean_within_class_std", 0)
    batch_f = diagnostics.get("batch_variability", {}).get("mean_f_statistic", 1)

    # Normalize to percentages (rough heuristic)
    tech_contrib = tech_cv
    bio_contrib = bio_std * 10  # Scale to comparable range
    batch_contrib = batch_f * 5

    total = tech_contrib + bio_contrib + batch_contrib + 1e-12

    variance_partition = {
        "technical_percent": float((tech_contrib / total) * 100),
        "biological_percent": float((bio_contrib / total) * 100),
        "batch_percent": float((batch_contrib / total) * 100),
        "note": "Heuristic partitioning; not rigorous variance components analysis.",
    }

    diagnostics["variance_partition"] = variance_partition

    return diagnostics
