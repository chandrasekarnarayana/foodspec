import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.io.exporters import to_hdf5, to_tidy_csv
from foodspec.io.loaders import load_folder, load_from_metadata_table


def _write_spectrum(path: Path, wavenumbers: np.ndarray, intensities: np.ndarray) -> None:
    data = np.column_stack([wavenumbers, intensities])
    np.savetxt(path, data)


def _create_sample_files(tmp_path: Path):
    w1 = np.array([600.0, 800.0, 1000.0])
    w2 = np.array([600.0, 800.0, 1000.0])
    w3 = np.array([600.0, 850.0, 1000.0])  # forces interpolation
    _write_spectrum(tmp_path / "s1.txt", w1, np.array([1.0, 2.0, 3.0]))
    _write_spectrum(tmp_path / "s2.txt", w2, np.array([1.5, 2.5, 3.5]))
    _write_spectrum(tmp_path / "s3.txt", w3, np.array([2.0, 3.0, 4.0]))


def test_load_folder_and_export(tmp_path: Path):
    _create_sample_files(tmp_path)
    meta = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "label": ["a", "b", "c"],
        }
    )
    meta_path = tmp_path / "metadata.csv"
    meta.to_csv(meta_path, index=False)

    ds = load_folder(tmp_path, metadata_csv=meta_path)
    assert isinstance(ds, FoodSpectrumSet)
    assert ds.x.shape == (3, 3)
    assert list(ds.metadata["label"]) == ["a", "b", "c"]

    tidy_path = tmp_path / "tidy.csv"
    to_tidy_csv(ds, tidy_path)
    tidy = pd.read_csv(tidy_path)
    assert tidy.shape[0] == ds.x.shape[0] * ds.x.shape[1]
    assert set(["sample_id", "label", "wavenumber", "intensity"]).issubset(tidy.columns)

    h5_path = tmp_path / "spectra.h5"
    h5 = pytest.importorskip("h5py")
    to_hdf5(ds, h5_path)
    with h5.File(h5_path, "r") as f:
        x_back = f["x"][...]
        w_back = f["wavenumbers"][...]
        meta_json = f["metadata_json"][()].decode()
        modality = f.attrs["modality"]
    # Use StringIO to avoid pandas warning on literal JSON (pandas 2.2+).
    meta_back = pd.read_json(io.StringIO(meta_json), orient="table")
    ds_back = FoodSpectrumSet(
        x=x_back, wavenumbers=w_back, metadata=meta_back, modality=modality
    )
    assert ds_back.x.shape == ds.x.shape
    assert ds_back.metadata.equals(ds.metadata)


def test_load_from_metadata_table(tmp_path: Path):
    _create_sample_files(tmp_path)
    table = pd.DataFrame(
        {
            "file_path": ["s1.txt", "s2.txt", "s3.txt"],
            "batch": [1, 1, 2],
        }
    )
    table_path = tmp_path / "table.csv"
    table.to_csv(table_path, index=False)

    ds = load_from_metadata_table(table_path, modality="raman")
    assert ds.x.shape == (3, 3)
    assert "batch" in ds.metadata.columns
    assert list(ds.metadata["sample_id"]) == ["s1", "s2", "s3"]
