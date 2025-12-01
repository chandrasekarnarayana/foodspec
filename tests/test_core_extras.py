import numpy as np
import pandas as pd
import pytest

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.core.hyperspectral import HyperSpectralCube


def _basic_fs():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    wn = np.array([100.0, 200.0, 300.0])
    meta = pd.DataFrame({"label": ["a", "b"]})
    return FoodSpectrumSet(x=x, wavenumbers=wn, metadata=meta, modality="raman")


def test_dataset_getitem_and_wide_dataframe():
    fs = _basic_fs()
    subset = fs[1]
    assert len(subset) == 1
    df = fs.to_wide_dataframe()
    assert "int_100.0" in df.columns
    assert df.shape[1] == fs.metadata.shape[1] + fs.x.shape[1]


def test_dataset_subset_filters_and_copy_shallow():
    fs = _basic_fs()
    filtered = fs.subset(by={"label": "a"})
    assert len(filtered) == 1
    shallow = fs.copy(deep=False)
    shallow.metadata.at[0, "label"] = "c"
    # shallow copy shares metadata
    assert fs.metadata.iloc[0]["label"] == "c"


def test_dataset_invalid_index_and_modality():
    fs = _basic_fs()
    with pytest.raises(IndexError):
        _ = fs[5]
    with pytest.raises(TypeError):
        _ = fs["bad"]  # type: ignore
    with pytest.raises(ValueError):
        FoodSpectrumSet(x=fs.x, wavenumbers=fs.wavenumbers, metadata=fs.metadata, modality="other")


def test_hyperspectral_index_errors_and_metadata_collision():
    cube_data = np.ones((2, 2, 3))
    wn = np.array([1.0, 2.0, 3.0])
    meta = pd.DataFrame({"sample_id": ["p0", "p1", "p2", "p3"], "row": [0, 0, 1, 1], "col": [0, 1, 0, 1]})
    cube = HyperSpectralCube(cube=cube_data, wavenumbers=wn, metadata=meta, image_shape=(2, 2))
    with pytest.raises(IndexError):
        cube.get_pixel_spectrum(5, 0)
    fs_pixels = cube.to_pixel_spectra()
    assert "row" in fs_pixels.metadata.columns and "col" in fs_pixels.metadata.columns
