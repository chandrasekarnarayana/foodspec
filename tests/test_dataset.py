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
        ]
    )
    wavenumbers = np.array([600.0, 800.0, 1000.0])
    metadata = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "group": ["train", "train", "test"],
        }
    )
    return FoodSpectrumSet(x=x, wavenumbers=wavenumbers, metadata=metadata, modality="raman")


def test_constructor_valid():
    ds = _make_dataset()
    assert len(ds) == 3
    wide = ds.to_wide_dataframe()
    assert set(["sample_id", "group", "int_600.0", "int_800.0", "int_1000.0"]).issubset(
        wide.columns
    )
    assert wide.shape == (3, 5)


def test_subset_by_metadata_and_indices():
    ds = _make_dataset()
    subset = ds.subset(by={"group": "train"}, indices=[0, 2])
    assert len(subset) == 1
    assert subset.metadata["sample_id"].iloc[0] == "a"
    # slice via __getitem__
    sliced = ds[1:]
    assert len(sliced) == 2
    assert list(sliced.metadata["sample_id"]) == ["b", "c"]


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

