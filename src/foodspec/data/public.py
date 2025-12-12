"""Public dataset loaders for foodspec.

These functions expect users to manually download open datasets to a known
location on disk. Each loader performs light parsing into a FoodSpectrumSet
and attaches basic metadata. They will raise clear errors with instructions
if the expected files are not present.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.validation import validate_public_evoo_sunflower, validate_spectrum_set

__all__ = [
    "load_public_mendeley_oils",
    "load_public_evoo_sunflower_raman",
    "load_public_ftir_oils",
]


def _default_root(subdir: str, root: Optional[PathLike]) -> Path:
    if root is None:
        return Path.home() / "foodspec_datasets" / subdir
    return Path(root)


def load_public_mendeley_oils(root: PathLike | None = None) -> FoodSpectrumSet:
    """Load Raman/FTIR edible oils from a public Mendeley dataset.

    Expected layout (user-provided):
    root/
      *.csv  (each CSV: first column wavenumbers, remaining columns spectra for a single modality)

    Metadata:
    - oil_type (from filename stem up to first underscore)
    - modality ("raman" or "ftir" inferred from filename if possible; default "raman")
    - source = "Mendeley"
    - doi = "10.17632/<dataset_id>" (replace with actual)

    Notes
    -----
    This loader does not download files. Download the dataset from Mendeley
    (e.g., https://data.mendeley.com/ with the DOI above) and place the CSVs under the default path:
    ``~/foodspec_datasets/mendeley_oils`` or pass ``root=...`` explicitly.
    """

    data_root = _default_root("mendeley_oils", root)
    csv_files = sorted(data_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {data_root}. "
            "Download the Mendeley oils dataset (see project README/DOI) and place CSVs here."
        )

    spectra_list = []
    labels = []
    modality_list = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        if df.shape[1] < 2:
            continue
        wn = df.iloc[:, 0].to_numpy(dtype=float)
        X = df.iloc[:, 1:].to_numpy(dtype=float).T  # each column is a spectrum
        oil_label = fpath.stem.split("_")[0]
        # Simple modality heuristic
        mod = "ftir" if "ftir" in fpath.stem.lower() else "raman"
        spectra_list.append((X, wn, oil_label, mod))
        labels.extend([oil_label] * X.shape[0])
        modality_list.extend([mod] * X.shape[0])

    if not spectra_list:
        raise FileNotFoundError(f"No usable spectra found in {data_root}.")

    # Assume all share a common axis (or first file as reference)
    X_concat = np.vstack([s[0] for s in spectra_list])
    wn_ref = spectra_list[0][1]
    metadata = pd.DataFrame(
        {
            "sample_id": [f"{lbl}_{i}" for i, lbl in enumerate(labels)],
            "oil_type": labels,
            "modality": modality_list,
            "source": ["Mendeley"] * len(labels),
            "doi": ["10.17632/<dataset_id>"] * len(labels),
        }
    )
    fs = FoodSpectrumSet(x=X_concat, wavenumbers=wn_ref, metadata=metadata, modality="raman")
    validate_spectrum_set(fs)
    return fs


def load_public_evoo_sunflower_raman(root: PathLike | None = None) -> FoodSpectrumSet:
    """Load a public French EVOO–sunflower Raman mixture dataset.

    Expected layout (user-provided):
    root/
      *.csv  (first column wavenumbers, second column intensity; one spectrum per file)

    Metadata:
    - mixture_fraction_evoo (parsed from filename if numeric; else NaN)
    - dataset_name = "EVOO-Sunflower Raman"
    - doi = "10.5281/zenodo.<record_id>" (replace with actual)
    - instrument = "<insert instrument name>"
    - oil_type (optional: derived from filename)
    - modality = "raman"

    Notes
    -----
    Download the dataset manually (e.g., from Zenodo with the DOI above) and place files under
    ``~/foodspec_datasets/evoo_sunflower_raman`` or pass ``root=...``.
    """

    data_root = _default_root("evoo_sunflower_raman", root)
    csv_files = sorted(data_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {data_root}. "
            "Download the EVOO–sunflower Raman dataset (see project README/DOI) and place CSVs here."
        )

    spectra = []
    labels = []
    fractions = []
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        if df.shape[1] < 2:
            continue
        wn = df.iloc[:, 0].to_numpy(dtype=float)
        intensity = df.iloc[:, 1].to_numpy(dtype=float)
        spectra.append(intensity)
        # parse mixture fraction from filename (e.g., evoo_70.csv -> 70)
        stem_parts = fpath.stem.split("_")
        frac = None
        for part in stem_parts:
            try:
                frac = float(part)
                if frac > 1:
                    frac = frac / 100.0
                break
            except ValueError:
                continue
        fractions.append(frac if frac is not None else np.nan)
        labels.append(fpath.stem)

    if not spectra:
        raise FileNotFoundError(f"No usable spectra found in {data_root}.")

    X = np.vstack(spectra)
    metadata = pd.DataFrame(
        {
            "sample_id": labels,
            "mixture_fraction_evoo": fractions,
            "dataset_name": ["EVOO-Sunflower Raman"] * len(labels),
            "doi": ["10.5281/zenodo.<record_id>"] * len(labels),
            "instrument": ["<insert instrument>"] * len(labels),
            "modality": ["raman"] * len(labels),
        }
    )
    fs = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=metadata, modality="raman")
    validate_spectrum_set(fs, allow_nan=True)
    validate_public_evoo_sunflower(fs)
    return fs


def load_public_ftir_oils(root: PathLike | None = None) -> FoodSpectrumSet:
    """Load a public FTIR edible-oil dataset.

    Expected layout (user-provided):
    root/
      *.csv (first column wavenumbers, remaining columns spectra) OR
      one spectrum per CSV (first col wn, second col intensity)

    Metadata:
    - oil_type (from filename prefix)
    - modality = "ftir"
    - source = "public"
    - doi = "10.5281/zenodo.<record_id>" (update to actual if known)

    Notes
    -----
    Download the dataset manually (e.g., from Zenodo or institutional repository) and place files under
    ``~/foodspec_datasets/ftir_oils`` or pass ``root=...``.
    """

    data_root = _default_root("ftir_oils", root)
    csv_files = sorted(data_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {data_root}. "
            "Download the public FTIR oils dataset (see project README/DOI) and place CSVs here."
        )

    spectra_list = []
    labels = []
    wn_ref = None
    for fpath in csv_files:
        df = pd.read_csv(fpath)
        if df.shape[1] < 2:
            continue
        oil_label = fpath.stem.split("_")[0]
        if df.shape[1] == 2:
            wn = df.iloc[:, 0].to_numpy(dtype=float)
            intensity = df.iloc[:, 1].to_numpy(dtype=float)
            spectra_list.append(intensity)
            labels.append(oil_label)
            if wn_ref is None:
                wn_ref = wn
        else:
            wn = df.iloc[:, 0].to_numpy(dtype=float)
            Xcols = df.iloc[:, 1:].to_numpy(dtype=float).T
            spectra_list.append(Xcols)
            labels.extend([oil_label] * Xcols.shape[0])
            if wn_ref is None:
                wn_ref = wn

    if not spectra_list:
        raise FileNotFoundError(f"No usable spectra found in {data_root}.")

    X_concat = np.vstack([s if s.ndim == 2 else s.reshape(1, -1) for s in spectra_list])
    metadata = pd.DataFrame(
        {
            "sample_id": [f"{lbl}_{i}" for i, lbl in enumerate(labels)],
            "oil_type": labels,
            "modality": ["ftir"] * len(labels),
            "source": ["public"] * len(labels),
            "doi": ["10.5281/zenodo.<record_id>"] * len(labels),
        }
    )
    fs = FoodSpectrumSet(x=X_concat, wavenumbers=wn_ref, metadata=metadata, modality="ftir")
    validate_spectrum_set(fs)
    return fs
