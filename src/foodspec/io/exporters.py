"""Export utilities for spectral datasets."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

from foodspec.data_objects.spectra_set import FoodSpectrumSet

__all__ = ["to_tidy_csv", "to_hdf5"]


def to_tidy_csv(spectra: FoodSpectrumSet, path: PathLike) -> None:
    """Export spectra to a tidy (long-form) CSV file.

    Produces columns `sample_id`, all metadata fields, `wavenumber`, and
    `intensity`.

    Args:
        spectra: Dataset to export.
        path: Output file path where the CSV will be written.
    """

    metadata = spectra.metadata.copy()
    if "sample_id" not in metadata.columns:
        metadata.insert(0, "sample_id", metadata.index.astype(str))

    n_samples, n_wavenumbers = spectra.x.shape
    repeated_metadata = {}
    for col in metadata.columns:
        repeated_metadata[col] = np.repeat(metadata[col].to_numpy(), n_wavenumbers)

    tidy = pd.DataFrame(repeated_metadata)
    tidy["wavenumber"] = np.tile(spectra.wavenumbers, n_samples)
    tidy["intensity"] = spectra.x.reshape(-1)

    # Order columns: sample_id, other metadata, wavenumber, intensity
    metadata_cols = list(metadata.columns)
    other_cols = [c for c in metadata_cols if c != "sample_id"]
    tidy = tidy[["sample_id", *other_cols, "wavenumber", "intensity"]]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(path, index=False)


def to_hdf5(spectra: FoodSpectrumSet, path: PathLike) -> None:
    """Persist spectra to an HDF5 file.

    Stores datasets `x`, `wavenumbers`, and `metadata_json` (serialized via
    `DataFrame.to_json`), plus the `modality` attribute.

    Args:
        spectra: Dataset to save.
        path: Target HDF5 file path.

    Raises:
        ImportError: If `h5py` is not installed.
    """

    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - tested via importorskip
        raise ImportError("h5py is required for HDF5 export.") from exc

    metadata_json = spectra.metadata.to_json(orient="table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=spectra.x)
        h5.create_dataset("wavenumbers", data=spectra.wavenumbers)
        h5.create_dataset("metadata_json", data=np.bytes_(metadata_json))
        h5.attrs["modality"] = spectra.modality
