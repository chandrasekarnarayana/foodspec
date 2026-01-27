from __future__ import annotations

"""
Core IO utilities.

Provides format detection and routing to the appropriate reader so inputs
are normalized into `FoodSpectrumSet` instances.
"""


import os
from pathlib import Path
from typing import Any

from foodspec.data_objects.spectra_set import FoodSpectrumSet
from foodspec.io import csv_import
from foodspec.io.text_formats import read_csv_folder, read_jcamp
from foodspec.io.validators import validate_spectrum_schema
from foodspec.io.vendor_formats import read_opus, read_spc


def detect_format(path: str | os.PathLike) -> str:
    """Detect input format based on path.

    Args:
        path: File or directory path.

    Returns:
        A short string key such as "csv", "folder_csv", "jcamp", "spc",
        "opus", "txt", or "unknown".
    """

    p = Path(path)
    if p.is_dir():
        return "folder_csv"
    ext = p.suffix.lower()
    if ext in {".csv"}:
        return "csv"
    if ext in {".txt"}:
        return "csv"
    if ext in {".jdx", ".dx"}:
        return "jcamp"
    if ext in {".spc"}:
        return "spc"
    if ext in {".0", ".1", ".opus"}:
        return "opus"
    return "unknown"


def _to_spectrum_set_from_df(df) -> FoodSpectrumSet:
    """Convert a DataFrame into a `FoodSpectrumSet`.

    Assumes the first column contains wavenumbers and the remaining columns
    are per-sample intensity values.

    Args:
        df: Input data where column 0 is wavenumbers and columns 1..N are
            intensity values for each sample.

    Returns:
        A `FoodSpectrumSet` with `x` shaped as (n_samples, n_wavenumbers)
        and the provided wavenumber axis.

    Raises:
        ValueError: If fewer than two columns are present (no intensity data).
    """

    if not isinstance(df, csv_import.pd.DataFrame):
        df = csv_import.pd.DataFrame(df)
    if df.shape[1] < 2:
        raise ValueError("Expected at least one intensity column alongside wavenumbers.")
    wavenumbers = df.iloc[:, 0].to_numpy(dtype=float)
    spectra = df.iloc[:, 1:].to_numpy(dtype=float).T  # samples x wn
    metadata = csv_import.pd.DataFrame({"sample_id": df.columns[1:]})
    fs = FoodSpectrumSet(x=spectra, wavenumbers=wavenumbers, metadata=metadata, modality="raman")
    validate_spectrum_schema(fs)
    return fs


def read_spectra(path: str | os.PathLike, format: str | None = None, **kwargs: Any) -> FoodSpectrumSet:
    """Read spectra from multiple possible formats into `FoodSpectrumSet`.

    Args:
        path: File or folder path.
        format: Optional override for the detected format. One of
            "csv", "folder_csv", "jcamp", "spc", "opus".
        **kwargs: Extra keyword arguments forwarded to the underlying loader.

    Returns:
        A `FoodSpectrumSet` loaded from the provided path.

    Raises:
        ValueError: If the format is unsupported or cannot be inferred.
    """

    fmt = format or detect_format(path)
    if fmt == "csv":
        # delegate to existing CSV import utility
        fs = csv_import.load_csv_spectra(path, format="wide")
        validate_spectrum_schema(fs)
        return fs
    if fmt == "folder_csv":
        df = read_csv_folder(path, **kwargs)
        return _to_spectrum_set_from_df(df)
    if fmt == "jcamp":
        fs = read_jcamp(path, **kwargs)
        validate_spectrum_schema(fs)
        return fs
    if fmt == "spc":
        fs = read_spc(path, **kwargs)
        validate_spectrum_schema(fs)
        return fs
    if fmt == "opus":
        fs = read_opus(path, **kwargs)
        validate_spectrum_schema(fs)
        return fs
    raise ValueError(f"Unsupported or unknown format for path: {path}")
