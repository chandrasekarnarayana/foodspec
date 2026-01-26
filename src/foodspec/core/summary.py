from __future__ import annotations
"""
Dataset summary metrics for food spectroscopy.

Provides at-a-glance quality indicators for spectral datasets:
- Samples per class distribution
- Basic statistical summaries (SNR, spectral range coverage)
- Missing data assessment
- Metadata completeness

**Key Assumptions:**
1. Dataset has valid metadata with class labels
2. Spectral data is numeric and finite (no NaN/inf)
3. Modality-specific quality thresholds apply (Raman vs FTIR)
4. Sufficient samples per class (typically ≥20) for robust analysis

**Typical Usage:**
    >>> from foodspec import FoodSpec
    >>> fs = FoodSpec("data.csv", modality="raman")
    >>> summary = fs.summarize_dataset()
    >>> print(summary["samples_per_class"])
    >>> print(summary["overall_quality_score"])
"""


import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from foodspec.data_objects.spectra_set import FoodSpectrumSet


def compute_samples_per_class(
    metadata: pd.DataFrame,
    label_column: str,
) -> Dict[str, Any]:
    """Compute sample count and distribution per class.

    Assumptions:
    - ``label_column`` exists in metadata
    - Labels are categorical or discrete
    - Labels may include missing values (reported)

    Args:
        metadata (pd.DataFrame): Dataset metadata.
        label_column (str): Column name with class labels.

    Returns:
        dict: Metrics including:
        - ``samples_per_class``: mapping class → count
        - ``total_samples``: total count
        - ``n_classes``: number of unique classes
        - ``min_class_size``: smallest class size
        - ``max_class_size``: largest class size
        - ``imbalance_ratio``: max/min class size
        - ``missing_labels``: count of missing labels

    Raises:
        ValueError: If ``label_column`` is missing from metadata.
    """
    if label_column not in metadata.columns:
        raise ValueError(f"label_column '{label_column}' not found in metadata.")

    labels = metadata[label_column]
    missing_count = labels.isna().sum()

    if missing_count > 0:
        warnings.warn(f"{missing_count} missing labels found in '{label_column}'.")

    # Count per class (excluding NaN)
    class_counts = labels.value_counts(dropna=True).to_dict()
    counts_array = np.array(list(class_counts.values()))

    if len(counts_array) == 0:
        return {
            "samples_per_class": {},
            "total_samples": int(len(metadata)),
            "n_classes": 0,
            "min_class_size": 0,
            "max_class_size": 0,
            "imbalance_ratio": float("inf"),
            "missing_labels": int(missing_count),
        }

    min_size = int(counts_array.min())
    max_size = int(counts_array.max())
    imbalance_ratio = max_size / (min_size + 1e-12)

    metrics = {
        "samples_per_class": {str(k): int(v) for k, v in class_counts.items()},
        "total_samples": int(len(metadata)),
        "n_classes": len(class_counts),
        "min_class_size": min_size,
        "max_class_size": max_size,
        "imbalance_ratio": float(imbalance_ratio),
        "missing_labels": int(missing_count),
    }

    return metrics


def compute_spectral_quality_metrics(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    modality: str = "raman",
) -> Dict[str, Any]:
    """Compute spectral data quality indicators.

    Assumptions:
    - Spectra are baseline-corrected or raw (not derivatives)
    - Wavenumbers are in cm⁻¹
    - NaN/inf values are handled and reported

    Args:
        spectra (np.ndarray): Shape (n_samples, n_wavenumbers) intensities.
        wavenumbers (np.ndarray): Shape (n_wavenumbers,) axis values.
        modality (str): Spectroscopy modality label.

    Returns:
        dict: Metrics including SNR estimate, spectral range, NaN/inf counts,
        negative rate, and intensity stats.
    """
    # Check for invalid values
    nan_count = np.isnan(spectra).sum()
    inf_count = np.isinf(spectra).sum()

    if nan_count > 0 or inf_count > 0:
        warnings.warn(f"Found {nan_count} NaN and {inf_count} inf values in spectra.")

    # Replace invalid with zeros for stats
    spectra_valid = np.where(np.isfinite(spectra), spectra, 0)

    # SNR estimate: mean / std per spectrum, then average
    means = spectra_valid.mean(axis=1)
    stds = spectra_valid.std(axis=1)
    snr_per_sample = means / (stds + 1e-12)
    mean_snr = snr_per_sample.mean()

    # Spectral range
    spectral_range = (float(wavenumbers.min()), float(wavenumbers.max()))

    # Negative intensities (may indicate baseline issues)
    negative_rate = (spectra_valid < 0).sum() / spectra_valid.size

    # Intensity stats
    intensity_min = float(spectra_valid.min())
    intensity_max = float(spectra_valid.max())
    intensity_mean = float(spectra_valid.mean())
    intensity_std = float(spectra_valid.std())

    metrics = {
        "mean_snr": float(mean_snr),
        "spectral_range": spectral_range,
        "n_wavenumbers": int(len(wavenumbers)),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "negative_intensity_rate": float(negative_rate),
        "intensity_range": {
            "min": intensity_min,
            "max": intensity_max,
            "mean": intensity_mean,
            "std": intensity_std,
        },
    }

    return metrics


def compute_metadata_completeness(
    metadata: pd.DataFrame,
    required_columns: Optional[list] = None,
) -> Dict[str, Any]:
    """Assess metadata completeness.

    Assumptions:
    - Metadata is a DataFrame
    - ``required_columns`` (if provided) are essential for analysis

    Args:
        metadata (pd.DataFrame): Dataset metadata.
        required_columns (list[str] | None): Columns that must be present and non-null.

    Returns:
        dict: Metrics including total columns, columns with missing, missing rate per column,
        required columns present, and overall completeness.
    """
    required_columns = required_columns or []

    total_cols = len(metadata.columns)
    missing_per_col = metadata.isna().sum()
    cols_with_missing = missing_per_col[missing_per_col > 0].index.tolist()

    missing_rate = (missing_per_col / len(metadata)).to_dict()

    # Check required columns
    required_present = True
    for col in required_columns:
        if col not in metadata.columns:
            required_present = False
            warnings.warn(f"Required column '{col}' not found in metadata.")
        elif metadata[col].isna().any():
            required_present = False
            warnings.warn(f"Required column '{col}' has missing values.")

    # Overall completeness
    overall_completeness = 1.0 - (metadata.isna().sum().sum() / metadata.size)

    metrics = {
        "total_columns": total_cols,
        "columns_with_missing": cols_with_missing,
        "missing_rate_per_column": {k: float(v) for k, v in missing_rate.items()},
        "required_columns_present": bool(required_present),
        "overall_completeness": float(overall_completeness),
    }

    return metrics


def summarize_dataset(
    dataset: FoodSpectrumSet,
    label_column: Optional[str] = None,
    required_metadata_columns: Optional[list] = None,
) -> Dict[str, Any]:
    """Comprehensive dataset summary for at-a-glance quality assessment.

    Workflow:
    1. Samples per class distribution
    2. Spectral quality metrics (SNR, range, NaN/inf)
    3. Metadata completeness

    Assumptions:
    - Dataset is a valid FoodSpectrumSet
    - ``label_column`` (if provided) is categorical
    - Spectral data is in standard format (no major preprocessing artifacts)

    Args:
        dataset (FoodSpectrumSet): Input dataset.
        label_column (str | None): Column with class labels for balance analysis.
        required_metadata_columns (list[str] | None): Columns required for workflows.

    Returns:
        dict: Summary sections including class distribution, spectral quality,
        metadata completeness, and dataset info.
    """
    summary = {}

    # Basic info
    summary["dataset_info"] = {
        "n_samples": len(dataset),
        "n_wavenumbers": dataset.x.shape[1],
        "modality": dataset.modality,
    }

    # Class distribution
    if label_column is not None:
        summary["class_distribution"] = compute_samples_per_class(dataset.metadata, label_column)

    # Spectral quality
    summary["spectral_quality"] = compute_spectral_quality_metrics(
        dataset.x,
        dataset.wavenumbers,
        modality=dataset.modality,
    )

    # Metadata completeness
    summary["metadata_completeness"] = compute_metadata_completeness(
        dataset.metadata,
        required_columns=required_metadata_columns,
    )

    return summary
