"""I/O loaders for spectral data."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet

__all__ = ["load_folder", "load_from_metadata_table"]


def load_folder(
    folder: PathLike,
    pattern: str = "*.txt",
    modality: str = "raman",
    metadata_csv: Optional[PathLike] = None,
    wavenumber_column: int = 0,
    intensity_columns: Optional[Sequence[int]] = None,
) -> FoodSpectrumSet:
    """Load spectra from a folder of text files into a ``FoodSpectrumSet``.

    Parameters
    ----------
    folder :
        Directory containing spectra files.
    pattern :
        Glob pattern for spectra files.
    modality :
        Spectroscopy modality string.
    metadata_csv :
        Optional CSV with a ``sample_id`` column used to merge metadata by file basename.
    wavenumber_column :
        Column index for wavenumbers in the spectra files.
    intensity_columns :
        Optional indices for intensity columns. If multiple are provided, their mean
        is taken. When omitted, all columns except ``wavenumber_column`` are used.

    Returns
    -------
    FoodSpectrumSet
        Loaded dataset with a common wavenumber axis.
    """

    folder_path = Path(folder)
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise ValueError(f"No files matching pattern '{pattern}' found in {folder_path}.")

    w_axes = []
    spectra = []
    sample_ids = []
    for file in files:
        wav, inten = _read_spectrum(
            file,
            wavenumber_column=wavenumber_column,
            intensity_columns=intensity_columns,
        )
        w_axes.append(wav)
        spectra.append(inten)
        sample_ids.append(file.stem)

    common_axis, stacked = _stack_spectra_on_common_axis(w_axes, spectra)

    metadata = pd.DataFrame({"sample_id": sample_ids})
    if metadata_csv is not None:
        meta_df = pd.read_csv(metadata_csv)
        if "sample_id" not in meta_df.columns:
            raise ValueError("metadata_csv must contain a 'sample_id' column.")
        metadata = metadata.merge(meta_df, on="sample_id", how="left")

    return FoodSpectrumSet(
        x=stacked,
        wavenumbers=common_axis,
        metadata=metadata,
        modality=modality,
    )


def load_from_metadata_table(
    metadata_csv: PathLike,
    modality: str = "raman",
    wavenumber_column: int = 0,
    intensity_columns: Optional[Sequence[int]] = None,
) -> FoodSpectrumSet:
    """Load spectra listed in a metadata table.

    Parameters
    ----------
    metadata_csv :
        CSV file with a ``file_path`` column and optional metadata columns.
    modality :
        Spectroscopy modality string.
    wavenumber_column :
        Column index for wavenumbers in the spectra files.
    intensity_columns :
        Optional indices for intensity columns. If multiple are provided, their mean
        is taken.

    Returns
    -------
    FoodSpectrumSet
        Loaded dataset with a common wavenumber axis.
    """

    table_path = Path(metadata_csv)
    table = pd.read_csv(table_path)
    if "file_path" not in table.columns:
        raise ValueError("metadata_csv must contain a 'file_path' column.")

    spectra = []
    w_axes = []
    sample_ids = []
    resolved_paths = []
    for file_entry in table["file_path"]:
        file_path = Path(file_entry)
        if not file_path.is_absolute():
            file_path = table_path.parent / file_path
        resolved_paths.append(file_path)
        wav, inten = _read_spectrum(
            file_path,
            wavenumber_column=wavenumber_column,
            intensity_columns=intensity_columns,
        )
        w_axes.append(wav)
        spectra.append(inten)
        sample_ids.append(file_path.stem)

    common_axis, stacked = _stack_spectra_on_common_axis(w_axes, spectra)

    metadata = table.drop(columns=["file_path"]).copy()
    if "sample_id" not in metadata.columns:
        metadata.insert(0, "sample_id", sample_ids)
    else:
        metadata = metadata.reset_index(drop=True)

    return FoodSpectrumSet(
        x=stacked,
        wavenumbers=common_axis,
        metadata=metadata,
        modality=modality,
    )


def _read_spectrum(
    file_path: PathLike,
    wavenumber_column: int,
    intensity_columns: Optional[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Read a single spectrum file."""

    data = np.loadtxt(file_path, ndmin=2)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"File {file_path} must contain at least two columns.")

    if intensity_columns is None:
        intensity_columns = [i for i in range(data.shape[1]) if i != wavenumber_column]
    if not intensity_columns:
        raise ValueError("No intensity columns specified.")

    wavenumbers = data[:, wavenumber_column]
    intensities = data[:, intensity_columns]
    if intensities.ndim == 2 and intensities.shape[1] > 1:
        intensities = np.nanmean(intensities, axis=1)
    elif intensities.ndim == 2:
        intensities = intensities[:, 0]

    # ensure sorted by wavenumber
    order = np.argsort(wavenumbers)
    wavenumbers = wavenumbers[order]
    intensities = intensities[order]
    return wavenumbers, intensities


def _stack_spectra_on_common_axis(
    w_axes: Sequence[np.ndarray], spectra: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Build common axis; interpolate spectra if needed."""

    reference = w_axes[0]
    identical = all(
        wav.shape == reference.shape and np.allclose(wav, reference)
        for wav in w_axes[1:]
    )
    if identical:
        stacked = np.vstack(spectra)
        return reference, stacked

    interpolated = []
    for wav, spec in zip(w_axes, spectra):
        interp = np.interp(reference, wav, spec, left=np.nan, right=np.nan)
        interpolated.append(interp)
    stacked = np.vstack(interpolated)
    return reference, stacked

