"""Data loading and validation for spectroscopy preprocessing.

Supports:
- Wide CSV: columns are wavenumbers; rows are samples
- Long CSV: columns include sample_id, x (wavenumber), y (intensity)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Modality = Literal["raman", "ftir", "ir", "nir", "unknown"]

# Metadata column names (optional)
METADATA_COLUMNS = {"batch", "stage", "instrument", "replicate", "matrix", "modality", "sample_id"}


@dataclass
class SpectraData:
    """Standard internal representation for spectral data."""

    X: np.ndarray  # (n_samples, n_features)
    wavenumbers: np.ndarray  # (n_features,) sorted
    metadata: pd.DataFrame  # (n_samples, metadata_cols)
    modality: Modality


def validate_modality(modality: str) -> Modality:
    """Validate and normalize modality string."""
    mod = str(modality).lower().strip()
    if mod in {"raman", "ftir", "ir", "nir"}:
        return mod
    if mod in {"raman ", " raman"}:
        return "raman"
    if mod in {"ftir", "ft-ir", "ft_ir"}:
        return "ftir"
    if mod in {"ir", "infrared"}:
        return "ir"
    return "unknown"


def _infer_modality_from_data(wavenumbers: np.ndarray) -> Modality:
    """Infer modality from wavenumber range."""
    if wavenumbers is None or len(wavenumbers) == 0:
        return "unknown"
    wmin, wmax = wavenumbers.min(), wavenumbers.max()
    # Raman typically 50-3500 cm-1
    if wmin < 100 and wmax > 3000:
        return "raman"
    # IR typically 400-4000 cm-1
    if wmin < 500 and wmax > 3000:
        return "ftir"
    return "unknown"


def load_csv(
    path: str | Path,
    format: Literal["wide", "long", "auto"] = "auto",
    metadata_cols: Optional[list[str]] = None,
    modality: Optional[str] = None,
) -> SpectraData:
    """Load CSV file into standard representation.

    Parameters
    ----------
    path : str | Path
        CSV file path.
    format : {'wide', 'long', 'auto'}
        Format: 'wide' = columns are wavenumbers; 'long' = columns include x, y.
    metadata_cols : list[str] | None
        Optional metadata column names to extract.
    modality : str | None
        Spectroscopy modality ('raman', 'ftir', 'ir', 'nir'). Auto-inferred if None.

    Returns
    -------
    SpectraData
        Normalized data in standard representation.
    """
    df = pd.read_csv(path)

    # Infer format if 'auto'
    if format == "auto":
        format = _infer_format(df)

    if format == "wide":
        X, wavenumbers, metadata = _load_wide_csv(df, metadata_cols=metadata_cols)
    elif format == "long":
        X, wavenumbers, metadata = _load_long_csv(df, metadata_cols=metadata_cols)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Infer modality
    if modality is None:
        modality = _infer_modality_from_data(wavenumbers)
    else:
        modality = validate_modality(modality)

    return SpectraData(X=X, wavenumbers=wavenumbers, metadata=metadata, modality=modality)


def _infer_format(df: pd.DataFrame) -> Literal["wide", "long"]:
    """Infer CSV format from structure."""
    # Look for long format indicators
    if "wavenumber" in df.columns or "x" in df.columns or "wavelength" in df.columns:
        if "intensity" in df.columns or "y" in df.columns:
            return "long"
    # Otherwise assume wide
    return "wide"


def _load_wide_csv(
    df: pd.DataFrame, metadata_cols: Optional[list[str]] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Parse wide format CSV (columns = wavenumbers).

    Example:
        sample_id, batch, 1000, 1001, ..., 3000
        oil_1,      B1,    0.234, 0.245, ..., 0.198
    """
    # Identify metadata columns (non-numeric or in METADATA_COLUMNS)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    # Extract metadata
    if non_numeric:
        meta_df = df[non_numeric].copy()
    else:
        meta_df = pd.DataFrame(index=df.index)

    # Extract wavenumbers and intensities
    X = df[numeric_cols].values.astype(float)
    wavenumbers = np.array([float(c) for c in numeric_cols])

    # Sort by wavenumber
    sort_idx = np.argsort(wavenumbers)
    X = X[:, sort_idx]
    wavenumbers = wavenumbers[sort_idx]

    return X, wavenumbers, meta_df


def _load_long_csv(
    df: pd.DataFrame, metadata_cols: Optional[list[str]] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Parse long format CSV (columns: sample_id, wavenumber, intensity, metadata).

    Example:
        sample_id, batch, wavenumber, intensity
        oil_1,      B1,    1000,       0.234
        oil_1,      B1,    1001,       0.245
    """
    # Identify key columns
    x_col = None
    for c in ["wavenumber", "x", "wavelength"]:
        if c in df.columns:
            x_col = c
            break

    y_col = None
    for c in ["intensity", "y", "value"]:
        if c in df.columns:
            y_col = c
            break

    sample_col = None
    for c in ["sample_id", "sample", "id"]:
        if c in df.columns:
            sample_col = c
            break

    if x_col is None or y_col is None or sample_col is None:
        raise ValueError(f"Cannot infer x, y, sample columns in long format. Available: {df.columns.tolist()}")

    # Pivot long to wide
    X_pivot = df.pivot_table(index=sample_col, columns=x_col, values=y_col, aggfunc="mean")
    X = X_pivot.values.astype(float)
    wavenumbers = X_pivot.columns.values.astype(float)

    # Sort by wavenumber
    sort_idx = np.argsort(wavenumbers)
    X = X[:, sort_idx]
    wavenumbers = wavenumbers[sort_idx]

    # Extract metadata (same columns for all rows per sample)
    meta_cols = [c for c in df.columns if c not in [x_col, y_col]]
    if meta_cols:
        meta_df = df.drop_duplicates(sample_col, keep="first")[meta_cols].reset_index(drop=True)
    else:
        meta_df = pd.DataFrame(index=range(X.shape[0]))

    return X, wavenumbers, meta_df


__all__ = [
    "SpectraData",
    "Modality",
    "METADATA_COLUMNS",
    "load_csv",
    "validate_modality",
]
