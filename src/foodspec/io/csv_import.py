from __future__ import annotations
"""CSV import utilities.

Converts public spectral datasets into `FoodSpectrumSet` instances.

Supported formats:
- "wide": one wavenumber column, one column per spectrum.
  Example columns: wavenumber, sample_001, sample_002, ...
- "long" (tidy): one row per `(sample_id, wavenumber)` with an intensity column.

The resulting `FoodSpectrumSet` can be used across FoodSpec workflows and
persisted using HDF5 utilities.
"""


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
    """Load spectra from a CSV file into a `FoodSpectrumSet`.

    Args:
        csv_path: Path to the CSV file.
        format: "wide" for one row per wavenumber and one column per spectrum,
            or "long" for one row per `(sample_id, wavenumber)` with an intensity
            column.
        wavenumber_column: Name of the wavenumber column (both formats).
        intensity_columns: For "wide" format, which columns contain intensities.
            If `None`, all non-wavenumber columns are treated as spectra.
        sample_id_column: For "long" format, column giving sample identifiers.
        intensity_column: For "long" format, column giving intensity values.
        label_column: Optional column name to copy into metadata (e.g., label).
        modality: Spectroscopy modality (e.g., "raman", "ftir").

    Returns:
        A `FoodSpectrumSet` ready for preprocessing and modeling.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or the format is invalid.
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
