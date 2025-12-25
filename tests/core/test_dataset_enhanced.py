import numpy as np
import pandas as pd
import pytest

from foodspec.core.dataset import FoodSpectrumSet


def _make_dataset() -> FoodSpectrumSet:
    x = np.vstack(
        [
            np.linspace(0.1, 1.0, 8),
            np.linspace(0.2, 1.1, 8),
            np.linspace(0.3, 1.2, 8),
            np.linspace(0.4, 1.3, 8),
        ]
    )
    wavenumbers = np.linspace(1000.0, 1200.0, 8)
    metadata = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "label": ["a", "a", "b", "b"],
            "group": ["train", "train", "test", "test"],
            "batch_id": [1, 1, 2, 2],
        }
    )
    return FoodSpectrumSet(
        x=x,
        wavenumbers=wavenumbers,
        metadata=metadata,
        modality="raman",
        label_col="label",
        group_col="group",
        batch_col="batch_id",
    )


def test_annotations_and_vectorized_ops():
    ds = _make_dataset()
    assert ds.labels.tolist() == ["a", "a", "b", "b"]
    assert ds.groups.tolist() == ["train", "train", "test", "test"]
    assert ds.batch_ids.tolist() == [1, 1, 2, 2]

    doubled = ds.scale(2.0)
    assert np.allclose(doubled.x, ds.x * 2)

    shifted = ds.offset(-0.1)
    assert np.allclose(shifted.x, ds.x - 0.1)

    inplace = ds.copy(deep=True)
    inplace.scale(0.5, inplace=True)
    assert np.allclose(inplace.x, ds.x * 0.5)

    augmented = ds.add_metadata_column("new_col", [10, 11, 12, 13])
    assert "new_col" in augmented.metadata.columns


def test_select_range_and_concat():
    ds = _make_dataset()
    window = ds.select_wavenumber_range(1050.0, 1150.0)
    assert window.x.shape[1] < ds.x.shape[1]
    assert np.all(window.wavenumbers >= 1050.0)
    assert np.all(window.wavenumbers <= 1150.0)

    combined = FoodSpectrumSet.concat([ds, ds])
    assert combined.x.shape[0] == 2 * ds.x.shape[0]
    assert combined.group_col == ds.group_col


def test_hdf5_roundtrip(tmp_path):
    pytest.importorskip("tables")
    ds = _make_dataset()
    path = tmp_path / "dataset.h5"
    ds.to_hdf5(path, key="ds")
    loaded = FoodSpectrumSet.from_hdf5(path, key="ds")
    assert np.allclose(loaded.x, ds.x)
    assert np.allclose(loaded.wavenumbers, ds.wavenumbers)
    assert list(loaded.metadata.columns) == list(ds.metadata.columns)
    assert loaded.label_col == "label"
    assert loaded.group_col == "group"
    assert loaded.batch_col == "batch_id"


def test_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    ds = _make_dataset()
    path = tmp_path / "dataset.parquet"
    ds.to_parquet(path)
    loaded = FoodSpectrumSet.from_parquet(path)
    assert np.allclose(loaded.x, ds.x)
    assert np.allclose(loaded.wavenumbers, ds.wavenumbers)
    assert loaded.label_col == "label"
    assert loaded.group_col == "group"
    assert loaded.batch_col == "batch_id"
