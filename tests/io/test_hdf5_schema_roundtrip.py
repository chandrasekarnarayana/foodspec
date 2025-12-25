from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from foodspec.core.spectral_dataset import SpectralDataset, HDF5_SCHEMA_VERSION


def test_hdf5_schema_roundtrip(tmp_path: Path):
    wn = np.array([1000.0, 1010.0])
    spectra = np.array([[1.0, 2.0], [3.0, 4.0]])
    meta = pd.DataFrame({"oil_type": ["A", "B"]})
    ds = SpectralDataset(wn, spectra, meta, instrument_meta={"instrument_id": "test"})
    path = tmp_path / "spec.h5"
    ds.save_hdf5(path)
    ds2 = SpectralDataset.from_hdf5(path)
    assert np.allclose(ds2.wavenumbers, wn)
    assert ds2.metadata.shape[0] == meta.shape[0]
    try:
        import h5py
    except ImportError:  # pragma: no cover
        return
    with h5py.File(path, "r") as f:
        assert f.attrs.get("foodspec_hdf5_schema_version") == HDF5_SCHEMA_VERSION


def test_hdf5_schema_version_guard(tmp_path: Path):
    wn = np.array([1000.0, 1010.0])
    spectra = np.array([[1.0, 2.0]])
    meta = pd.DataFrame({"oil_type": ["A"]})
    ds = SpectralDataset(wn, spectra, meta)
    path = tmp_path / "spec_future.h5"
    ds.save_hdf5(path)

    try:
        import h5py
    except ImportError:  # pragma: no cover - optional
        pytest.skip("h5py not installed")

    with h5py.File(path, "r+") as f:
        f.attrs["foodspec_hdf5_schema_version"] = "2.0"

    with pytest.raises(ValueError, match="Incompatible HDF5"):
        SpectralDataset.from_hdf5(path)

    # allow_future=True bypasses all guard checks
    ds_ok = SpectralDataset.from_hdf5(path, allow_future=True)
    assert isinstance(ds_ok, SpectralDataset)
