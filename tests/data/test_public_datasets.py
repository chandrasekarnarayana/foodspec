from pathlib import Path

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.data.public import (
    load_public_evoo_sunflower_raman,
    load_public_ftir_oils,
    load_public_mendeley_oils,
)
from foodspec.validation import validate_public_evoo_sunflower


def _write_csv(path: Path, wn: np.ndarray, spectra: np.ndarray):
    df = pd.DataFrame(np.column_stack([wn, spectra]))
    df.to_csv(path, index=False)


def test_load_public_mendeley_oils(tmp_path):
    wn = np.linspace(1000, 1010, 5)
    # Two spectra in columns after wavenumbers
    spectra = np.column_stack([wn, np.random.rand(5), np.random.rand(5)])
    _write_csv(tmp_path / "olive_raman.csv", wn, spectra[:, 1:])
    ds = load_public_mendeley_oils(root=tmp_path)
    assert isinstance(ds, FoodSpectrumSet)
    assert ds.modality == "raman"
    assert "oil_type" in ds.metadata
    assert len(ds) == 2
    assert len(ds.metadata) == ds.x.shape[0]


def test_load_public_evoo_sunflower_raman(tmp_path):
    wn = np.linspace(1000, 1010, 5)
    # One spectrum per file: first col wn, second intensity
    spectra = np.column_stack([wn, np.random.rand(5)])
    _write_csv(tmp_path / "evoo_70.csv", wn, spectra[:, 1:])
    ds = load_public_evoo_sunflower_raman(root=tmp_path)
    assert isinstance(ds, FoodSpectrumSet)
    assert ds.modality == "raman"
    assert "mixture_fraction_evoo" in ds.metadata
    assert len(ds) == 1
    assert len(ds.metadata) == ds.x.shape[0]
    # Validation should pass for in-range fractions
    validate_public_evoo_sunflower(ds)


def test_load_public_ftir_oils(tmp_path):
    wn = np.linspace(1000, 1010, 5)
    # Two spectra in columns after wavenumbers
    spectra = np.column_stack([wn, np.random.rand(5), np.random.rand(5)])
    _write_csv(tmp_path / "coconut_ftir.csv", wn, spectra[:, 1:])
    ds = load_public_ftir_oils(root=tmp_path)
    assert isinstance(ds, FoodSpectrumSet)
    assert ds.modality == "ftir"
    assert "oil_type" in ds.metadata
    assert len(ds) == 2
    assert len(ds.metadata) == ds.x.shape[0]
