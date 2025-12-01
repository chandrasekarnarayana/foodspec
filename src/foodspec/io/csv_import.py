"""CSV import utilities for converting public spectral datasets into FoodSpectrumSet.

Supported formats
-----------------
1) 'wide' format (one wavenumber column, one column per spectrum):
   wavenumber, sample_001, sample_002, ...
   500,       123.4,      98.1,       ...
   502,       124.0,      99.2,       ...

2) 'long' / tidy format (one row per (sample, wavenumber)):
   sample_id, wavenumber, intensity, [label], [modality], [any other metadata...]

The loader returns a FoodSpectrumSet, which can be saved as an HDF5 library via
create_library() and used by all foodspec workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.validation import validate_spectrum_set

__all__ = ["load_csv_spectra"]


def load_csv_spectra(
    csv_path: str | Path,
    format: str = "wide",
    *,
    wavenumber_column: str = "wavenumber",
    intensity_columns: Optional[Iterable[str]] = None,
    sample_id_column: str = "sample_id",
    intensity_column: str = "intensity",
    label_column: Optional[str] = None,
    modality: str = "raman",
) -> FoodSpectrumSet:
    """
    Load spectra from a CSV file into a FoodSpectrumSet.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    format:
        'wide'  – one row per wavenumber, one column per sample spectrum.
        'long'  – one row per (sample, wavenumber) with an intensity column.
    wavenumber_column:
        Name of the wavenumber column (both formats).
    intensity_columns:
        For 'wide' format: which columns contain intensities.
        If None, all non-wavenumber columns are treated as spectra.
    sample_id_column:
        For 'long' format: column giving sample identifiers.
    intensity_column:
        For 'long' format: column giving intensity values.
    label_column:
        Optional column to copy into metadata (e.g., oil_type).
    modality:
        'raman', 'ftir', etc. Used to tag the FoodSpectrumSet.

    Returns
    -------
    FoodSpectrumSet
        Spectral dataset ready for preprocessing and modeling.
    """

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if format.lower() == "wide":
        if wavenumber_column not in df.columns:
            raise ValueError(f"Expected wavenumber column '{wavenumber_column}' in CSV.")
        wn = df[wavenumber_column].to_numpy(dtype=float)

        if intensity_columns is None:
            intensity_columns = [c for c in df.columns if c != wavenumber_column]
        intensity_columns = list(intensity_columns)
        if not intensity_columns:
            raise ValueError("No intensity columns found for 'wide' CSV format.")

        # Transpose so rows = samples, cols = wavenumbers
        spectra = df[intensity_columns].to_numpy(dtype=float).T  # shape: (n_samples, n_wn)
        metadata = pd.DataFrame({"sample_id": intensity_columns})
        if label_column and label_column in df.columns:
            metadata[label_column] = np.nan

    elif format.lower() == "long":
        for col in (sample_id_column, wavenumber_column, intensity_column):
            if col not in df.columns:
                raise ValueError(f"Expected column '{col}' in 'long' CSV format.")

        wn = df[wavenumber_column].drop_duplicates().sort_values().to_numpy(dtype=float)

        pivot = df.pivot_table(
            index=sample_id_column,
            columns=wavenumber_column,
            values=intensity_column,
        )
        pivot = pivot.reindex(columns=wn)
        spectra = pivot.to_numpy(dtype=float)
        metadata = pd.DataFrame({sample_id_column: pivot.index.to_list()})

        if label_column and label_column in df.columns:
            labels = (
                df[[sample_id_column, label_column]]
                .drop_duplicates(subset=[sample_id_column])
                .set_index(sample_id_column)[label_column]
            )
            metadata[label_column] = metadata[sample_id_column].map(labels)
    else:
        raise ValueError("format must be 'wide' or 'long'.")

    ds = FoodSpectrumSet(
        x=spectra,
        wavenumbers=wn,
        metadata=metadata,
        modality=modality,
    )
    validate_spectrum_set(ds)
    return ds
