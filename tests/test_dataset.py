import numpy as np
import pandas as pd
import pytest

from foodspec.core.dataset import FoodSpectrumSet


def _make_dataset() -> FoodSpectrumSet:
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
        ]
    )
    wavenumbers = np.array([600.0, 800.0, 1000.0])
    metadata = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "group": ["train", "train", "test", "test"],
        }
    )
    return FoodSpectrumSet(x=x, wavenumbers=wavenumbers, metadata=metadata, modality="raman")


def test_constructor_valid():
    ds = _make_dataset()
    assert len(ds) == 4
    wide = ds.to_wide_dataframe()
    assert set(["sample_id", "group", "int_600.0", "int_800.0", "int_1000.0"]).issubset(wide.columns)
    assert wide.shape == (4, 5)


def test_subset_by_metadata_and_indices():
    ds = _make_dataset()
    subset = ds.subset(by={"group": "train"}, indices=[0, 2])
    assert len(subset) == 1
    assert subset.metadata["sample_id"].iloc[0] == "a"
    # slice via __getitem__
    sliced = ds[1:]
    assert len(sliced) == 3
    assert list(sliced.metadata["sample_id"]) == ["b", "c", "d"]


def test_invalid_shapes_raise_error():
    x = np.array([[1.0, 2.0]])
    wavenumbers = np.array([600.0, 800.0])
    metadata = pd.DataFrame({"sample_id": ["a"]})
    with pytest.raises(ValueError):
        FoodSpectrumSet(x=x, wavenumbers=wavenumbers, metadata=metadata, modality="raman")
    with pytest.raises(ValueError):
        FoodSpectrumSet(
            x=np.array([1.0, 2.0]),  # 1D instead of 2D
            wavenumbers=np.array([600.0, 800.0]),
            metadata=metadata,
            modality="raman",
        )
    with pytest.raises(ValueError):
        FoodSpectrumSet(
            x=np.array([[1.0, 2.0]]),
            wavenumbers=np.array([600.0]),
            metadata=metadata,
            modality="uv-vis",  # invalid modality
        )


def test_to_X_y_happy_path():
    ds = _make_dataset()
    X, y = ds.to_X_y("group")
    assert X.shape == (4, 3)
    assert y.tolist() == ["train", "train", "test", "test"]


def test_to_X_y_missing_column_raises():
    ds = _make_dataset()
    with pytest.raises(ValueError):
        ds.to_X_y("missing")


def test_train_test_split_stratified():
    ds = _make_dataset()
    train_ds, test_ds = ds.train_test_split(target_col="group", test_size=0.5, stratify=True, random_state=0)
    assert isinstance(train_ds, FoodSpectrumSet)
    assert isinstance(test_ds, FoodSpectrumSet)
    assert np.allclose(train_ds.wavenumbers, ds.wavenumbers)
    assert np.allclose(test_ds.wavenumbers, ds.wavenumbers)
    assert len(train_ds.metadata) == train_ds.x.shape[0]
    assert len(test_ds.metadata) == test_ds.x.shape[0]
