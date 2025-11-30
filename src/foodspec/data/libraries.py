"""Local spectral library utilities."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Literal
import io

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.features.fingerprint import (
    correlation_similarity_matrix,
    cosine_similarity_matrix,
)

__all__ = ["create_library", "load_library", "search_library_fingerprint"]


def create_library(path: PathLike, spectra: FoodSpectrumSet) -> None:
    """Persist a spectral library to HDF5."""

    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
        raise ImportError("h5py is required to create libraries.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = spectra.metadata.to_json(orient="table")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=spectra.x)
        h5.create_dataset("wavenumbers", data=spectra.wavenumbers)
        h5.create_dataset("metadata_json", data=np.bytes_(metadata_json))
        h5.attrs["modality"] = spectra.modality


def load_library(path: PathLike) -> FoodSpectrumSet:
    """Load a spectral library from HDF5."""

    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
        raise ImportError("h5py is required to load libraries.") from exc

    with h5py.File(path, "r") as h5:
        x = h5["x"][...]
        wavenumbers = h5["wavenumbers"][...]
        metadata_json = h5["metadata_json"][()].decode()
        modality = h5.attrs["modality"]
    # Explicit orient and dtype backend avoid pandas warnings on literal JSON (pandas 2.2+).
    metadata = pd.read_json(io.StringIO(metadata_json), orient="table")
    return FoodSpectrumSet(
        x=x, wavenumbers=wavenumbers, metadata=metadata, modality=modality
    )


def search_library_fingerprint(
    library: FoodSpectrumSet,
    query: FoodSpectrumSet,
    metric: Literal["cosine", "correlation"] = "cosine",
) -> pd.DataFrame:
    """Compute similarity scores between query spectra and a library."""

    if library.wavenumbers.shape != query.wavenumbers.shape or not np.allclose(
        library.wavenumbers, query.wavenumbers
    ):
        raise ValueError("Library and query must share identical wavenumber axes.")

    if metric == "cosine":
        sims = cosine_similarity_matrix(query.x, library.x)
    elif metric == "correlation":
        sims = correlation_similarity_matrix(query.x, library.x)
    else:
        raise ValueError("metric must be 'cosine' or 'correlation'.")

    lib_ids = library.metadata["sample_id"].tolist()
    query_ids = query.metadata["sample_id"].tolist()
    return pd.DataFrame(sims, index=query_ids, columns=lib_ids)
