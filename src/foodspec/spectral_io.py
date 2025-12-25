"""
Simple loaders/savers for vendor-neutral spectra.
Includes stubs for vendor formats (OPUS, WiRE, ENVI) that can be upgraded later.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from foodspec.core.spectral_dataset import SpectralDataset


def load_any_spectra(path: Union[str, Path], format_hint: Optional[str] = None) -> SpectralDataset:
    p = Path(path)
    if p.is_dir():
        raise ValueError("Directory loading not implemented for spectra; provide a file.")
    fmt = detect_format(p) if format_hint is None else format_hint
    if fmt == "opus":
        return load_opus(p)
    if fmt == "wire":
        return load_wire(p)
    if fmt == "envi":
        return load_envi(p)
    df = pd.read_csv(p)

    # Heuristic: numeric columns are wavenumbers, others metadata
    wavenumber_cols: List[str] = []
    meta_cols: List[str] = []
    for col in df.columns:
        try:
            float(col)
            wavenumber_cols.append(col)
        except Exception:
            meta_cols.append(col)
    wn = np.array([float(c) for c in wavenumber_cols])
    spectra = df[wavenumber_cols].to_numpy(dtype=float)
    metadata = df[meta_cols].reset_index(drop=True)
    return SpectralDataset(wavenumbers=wn, spectra=spectra, metadata=metadata)


def load_foodspec_csv(path: Union[str, Path], metadata_cols: Optional[List[str]] = None) -> SpectralDataset:
    df = pd.read_csv(path)
    meta_cols = metadata_cols or [c for c in df.columns if not str(c).replace(".", "", 1).isdigit()]
    spec_cols = [c for c in df.columns if c not in meta_cols]
    wn = np.array([float(c) for c in spec_cols])
    spectra = df[spec_cols].to_numpy(dtype=float)
    metadata = df[meta_cols].reset_index(drop=True)
    return SpectralDataset(wavenumbers=wn, spectra=spectra, metadata=metadata)


def load_opus(path: Union[str, Path]) -> SpectralDataset:
    """
    Stub OPUS loader (expects ASCII export).
    Future: replace with real OPUS parser.
    """
    try:
        ds = load_foodspec_csv(path)
        if ds.spectra.shape[1] == 0:
            raise ValueError("No numeric wavenumber columns found.")
        return ds
    except Exception as exc:
        raise ValueError(
            f"This file looks like OPUS but parsing failed. Ensure you export as ASCII/CSV. Details: {exc}"
        )


def load_wire(path: Union[str, Path]) -> SpectralDataset:
    """
    Stub Renishaw WiRE loader (expects ASCII export).
    """
    try:
        ds = load_foodspec_csv(path)
        if ds.spectra.shape[1] == 0:
            raise ValueError("No numeric wavenumber columns found.")
        return ds
    except Exception as exc:
        raise ValueError(f"This file looks like a WiRE export but parsing failed. Try exporting as CSV. Details: {exc}")


def load_envi(path_hdr: Union[str, Path], path_dat: Optional[Union[str, Path]] = None) -> SpectralDataset:
    """
    Stub ENVI loader for spectral tables (not HSI).
    """
    try:
        ds = load_foodspec_csv(path_hdr)
        if ds.spectra.shape[1] == 0:
            raise ValueError("No numeric wavenumber columns found.")
        return ds
    except Exception as exc:
        raise ValueError(f"This file looks like ENVI but parsing failed. Provide ASCII/CSV export. Details: {exc}")


def detect_format(path: Union[str, Path]) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".0", ".dat", ".opus"}:
        return "opus"
    if ext in {".spc", ".wdf", ".txt"} and "wire" in p.name.lower():
        return "wire"
    if ext in {".hdr", ".envi"}:
        return "envi"
    if ext in {".csv", ".tsv", ".txt"}:
        return "csv"
    return "csv"


def align_wavenumbers(
    datasets: List[SpectralDataset],
    target_grid: Optional[np.ndarray] = None,
    method: str = "interp",
) -> List[SpectralDataset]:
    """
    Align multiple datasets to a common wavenumber grid via interpolation.
    """
    if target_grid is None:
        target_grid = max(datasets, key=lambda d: len(d.wavenumbers)).wavenumbers
    out = []
    for ds in datasets:
        wn = ds.wavenumbers
        X = ds.spectra
        if method == "interp":
            interp = np.vstack([np.interp(target_grid, wn, row) for row in X])
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        ds_h = ds.copy()
        ds_h.wavenumbers = target_grid.copy()
        ds_h.spectra = interp
        harmonization_params = {"method": method, "len_grid": len(target_grid)}
        ds_h.logs.append(f"aligned to grid len={len(target_grid)}")
        ds_h.history.append({"step": "align_wavenumbers", "params": harmonization_params})
        ds_h.instrument_meta["harmonization_params"] = harmonization_params
        out.append(ds_h)
    return out


__all__ = [
    "load_any_spectra",
    "load_foodspec_csv",
    "load_opus",
    "load_wire",
    "load_envi",
    "align_wavenumbers",
    "detect_format",
]
