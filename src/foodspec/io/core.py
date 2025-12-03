"""
Core IO utilities: detect format and route to appropriate loader.
"""

from __future__ import annotations

import os
from pathlib import Path

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.io import csv_import
from foodspec.io.text_formats import read_csv_folder, read_jcamp
from foodspec.io.vendor_formats import read_opus, read_spc


def detect_format(path: str | os.PathLike) -> str:
    """
    Detect input format based on extension or folder.

    Returns a string key such as: 'csv', 'folder_csv', 'jcamp', 'spc', 'opus', 'txt'.
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
    """
    Build FoodSpectrumSet from a DataFrame with wavenumber plus intensity columns.

    Assumes first column is wavenumber and remaining columns are samples.
    """

    if not isinstance(df, csv_import.pd.DataFrame):
        df = csv_import.pd.DataFrame(df)
    if df.shape[1] < 2:
        raise ValueError("Expected at least one intensity column alongside wavenumbers.")
    wavenumbers = df.iloc[:, 0].to_numpy(dtype=float)
    spectra = df.iloc[:, 1:].to_numpy(dtype=float).T  # samples x wn
    metadata = csv_import.pd.DataFrame({"sample_id": df.columns[1:]})
    return FoodSpectrumSet(x=spectra, wavenumbers=wavenumbers, metadata=metadata, modality="raman")


def read_spectra(path: str | os.PathLike, format: str | None = None, **kwargs) -> FoodSpectrumSet:
    """
    High-level loader that normalizes various formats into FoodSpectrumSet.

    Parameters
    ----------
    path : str or Path
        File or folder path.
    format : str, optional
        Override detected format ('csv', 'folder_csv', 'jcamp', 'spc', 'opus').

    Returns
    -------
    FoodSpectrumSet
    """

    fmt = format or detect_format(path)
    if fmt == "csv":
        # delegate to existing CSV import utility
        return csv_import.load_csv_spectra(path, format="wide")
    if fmt == "folder_csv":
        df = read_csv_folder(path, **kwargs)
        return _to_spectrum_set_from_df(df)
    if fmt == "jcamp":
        fs = read_jcamp(path, **kwargs)
        return fs
    if fmt == "spc":
        return read_spc(path, **kwargs)
    if fmt == "opus":
        return read_opus(path, **kwargs)
    raise ValueError(f"Unsupported or unknown format for path: {path}")
