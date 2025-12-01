import numpy as np
import pandas as pd
import pytest

from foodspec.core.dataset import FoodSpectrumSet


def test_subset_invalid_metadata_key():
    ds = FoodSpectrumSet(
        x=np.ones((2, 3)),
        wavenumbers=np.array([1.0, 2.0, 3.0]),
        metadata=pd.DataFrame({"sample_id": ["a", "b"], "group": ["g1", "g2"]}),
        modality="raman",
    )
    with pytest.raises(ValueError):
        ds.subset(by={"missing": "x"})


def test_copy_shallow_vs_deep():
    ds = FoodSpectrumSet(
        x=np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
        wavenumbers=np.array([1.0, 2.0, 3.0]),
        metadata=pd.DataFrame({"sample_id": ["a", "b"], "group": ["g1", "g2"]}),
        modality="raman",
    )
    shallow = ds.copy(deep=False)
    deep = ds.copy(deep=True)
    ds.x[0, 0] = 9.0
    # shallow shares data
    assert shallow.x[0, 0] == 9.0
    # deep copy unaffected
    assert deep.x[0, 0] == 1.0


def test_non_monotonic_wavenumbers_error():
    with pytest.raises(ValueError):
        FoodSpectrumSet(
            x=np.ones((2, 2)),
            wavenumbers=np.array([2.0, 1.0]),
            metadata=pd.DataFrame({"sample_id": ["a", "b"]}),
            modality="raman",
        )


def test_normalize_index_invalid_type():
    ds = FoodSpectrumSet(
        x=np.ones((2, 3)),
        wavenumbers=np.array([1.0, 2.0, 3.0]),
        metadata=pd.DataFrame({"sample_id": ["a", "b"], "group": ["g1", "g2"]}),
        modality="raman",
    )
    with pytest.raises(TypeError):
        _ = ds["bad"]  # type: ignore
