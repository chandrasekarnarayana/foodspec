"""
Text-based format readers (CSV/TXT, JCAMP-DX).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.io.csv_import import load_csv_spectra


def read_csv_table(path: str | Path, format: Literal["wide", "long"] = "wide", **kwargs) -> FoodSpectrumSet:
    """
    Read a CSV file (wide or long) into a FoodSpectrumSet using load_csv_spectra.
    """

    return load_csv_spectra(path, format=format, **kwargs)


def read_csv_folder(path: str | Path, pattern: str = "*.csv", modality: str = "unknown") -> pd.DataFrame:
    """
    Read a folder of CSV files (one spectrum per file) into a wide DataFrame.

    Assumes each file has two columns: wavenumber, intensity.
    """

    path = Path(path)
    records = []
    wn_ref: Optional[np.ndarray] = None
    for csv_file in sorted(path.glob(pattern)):
        df = pd.read_csv(csv_file)
        if df.shape[1] < 2:
            raise ValueError(f"Expected at least two columns (wavenumber, intensity) in {csv_file}")
        wn = df.iloc[:, 0].to_numpy(dtype=float)
        intensity = df.iloc[:, 1].to_numpy(dtype=float)
        if wn_ref is None:
            wn_ref = wn
        else:
            if len(wn_ref) != len(wn):
                raise ValueError("Inconsistent wavenumber axes across folder files.")
        records.append((csv_file.stem, intensity))
    if wn_ref is None:
        raise ValueError("No matching CSV files found.")
    wide = pd.DataFrame({"wavenumber": wn_ref})
    for name, vals in records:
        wide[name] = vals
    return wide


def read_jcamp(path: str | Path, modality: str = "raman") -> FoodSpectrumSet:
    """
    Read a JCAMP-DX file (.jdx, .dx) into a FoodSpectrumSet.

    Minimal parser: extracts numeric pairs (wavenumber, intensity), ignoring header tags.
    """

    path = Path(path)
    wn = []
    inten = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    wn.append(x)
                    inten.append(y)
                except ValueError:
                    continue
    if not wn:
        raise ValueError(f"No spectral data found in JCAMP file: {path}")
    wn_arr = np.asarray(wn, dtype=float)
    inten_arr = np.asarray(inten, dtype=float)[np.newaxis, :]  # single sample
    metadata = pd.DataFrame({"sample_id": [path.stem]})
    return FoodSpectrumSet(x=inten_arr, wavenumbers=wn_arr, metadata=metadata, modality=modality)
