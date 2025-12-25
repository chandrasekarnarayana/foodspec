"""Tests for FoodSpectrumSet methods to increase coverage."""

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet


def test_select_wavenumber_range():
    """Test wavenumber range selection."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(10, 100)
    meta = pd.DataFrame({"id": range(10)})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    ds_subset = ds.select_wavenumber_range(700, 1200)

    assert ds_subset.wavenumbers.min() >= 700
    assert ds_subset.wavenumbers.max() <= 1200
    assert ds_subset.x.shape[1] < ds.x.shape[1]


def test_apply_function():
    """Test apply method."""
    wn = np.linspace(500, 1500, 50)
    x = np.random.randn(5, 50)
    meta = pd.DataFrame({"id": range(5)})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    ds_transformed = ds.apply(lambda spec: spec * 2)

    assert np.allclose(ds_transformed.x, ds.x * 2)


def test_scale_and_offset():
    """Test scale and offset operations."""
    wn = np.linspace(500, 1500, 50)
    x = np.random.randn(5, 50) + 10
    meta = pd.DataFrame({"id": range(5)})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    ds_scaled = ds.scale(2.0)
    ds_offset = ds.offset(5.0)

    assert np.allclose(ds_scaled.x, ds.x * 2.0)
    assert np.allclose(ds_offset.x, ds.x + 5.0)


def test_add_metadata_column():
    """Test adding metadata column."""
    wn = np.linspace(500, 1500, 50)
    x = np.random.randn(5, 50)
    meta = pd.DataFrame({"id": range(5)})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    new_col = ["X", "Y", "X", "Y", "Z"]
    ds_updated = ds.add_metadata_column("category", new_col)

    assert "category" in ds_updated.metadata.columns
    assert list(ds_updated.metadata["category"]) == new_col


def test_to_X_y():
    """Test conversion to X, y arrays."""
    wn = np.linspace(500, 1500, 50)
    x = np.random.randn(8, 50)
    meta = pd.DataFrame({"label": [0, 1, 0, 1, 0, 1, 0, 1]})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    X, y = ds.to_X_y(target_col="label")

    assert X.shape == (8, 50)
    assert len(y) == 8
    assert np.array_equal(y, meta["label"].values)


def test_train_test_split():
    """Test train/test split functionality."""
    wn = np.linspace(500, 1500, 50)
    x = np.random.randn(20, 50)
    meta = pd.DataFrame({"class": ["A"] * 10 + ["B"] * 10})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    train, test = ds.train_test_split(target_col="class", test_size=0.3, random_state=42)

    assert train.x.shape[0] + test.x.shape[0] == 20
    assert train.x.shape[1] == 50
    assert test.x.shape[1] == 50


def test_getitem_integer():
    """Test integer indexing."""
    wn = np.linspace(500, 1500, 50)
    x = np.random.randn(10, 50)
    meta = pd.DataFrame({"id": range(10)})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")

    ds_single = ds[3]
    assert ds_single.x.shape == (1, 50)
    assert len(ds_single.metadata) == 1


def test_getitem_slice():
    """Test slice indexing."""
    wn = np.linspace(500, 1500, 50)
    x = np.random.randn(10, 50)
    meta = pd.DataFrame({"id": range(10)})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")

    ds_slice = ds[2:5]
    assert ds_slice.x.shape == (3, 50)
    assert len(ds_slice.metadata) == 3


def test_concat_spectrum_sets():
    """Test concatenation of spectrum sets."""
    wn = np.linspace(500, 1500, 50)
    x1 = np.random.randn(5, 50)
    x2 = np.random.randn(3, 50)
    meta1 = pd.DataFrame({"id": range(5)})
    meta2 = pd.DataFrame({"id": range(5, 8)})

    ds1 = FoodSpectrumSet(x1, wn, meta1, modality="raman")
    ds2 = FoodSpectrumSet(x2, wn, meta2, modality="raman")

    ds_combined = FoodSpectrumSet.concat([ds1, ds2])

    assert ds_combined.x.shape[0] == 8
    assert len(ds_combined.metadata) == 8


def test_with_annotations():
    """Test updating annotations."""
    wn = np.linspace(500, 1500, 50)
    x = np.random.randn(5, 50)
    meta = pd.DataFrame({"id": range(5), "label": ["A", "B", "A", "B", "C"]})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    ds_annotated = ds.with_annotations(label_col="label")

    assert ds_annotated.label_col == "label"


def test_to_wide_dataframe():
    """Test conversion to wide dataframe."""
    wn = np.array([800.0, 900.0, 1000.0])
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    meta = pd.DataFrame({"id": [1, 2]})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    df = ds.to_wide_dataframe()

    assert df.shape[0] == 2
    assert len(df.columns) == 4  # id + 3 wavenumber columns
