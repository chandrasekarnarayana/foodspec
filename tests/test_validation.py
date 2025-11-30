import numpy as np
import pandas as pd
import pytest

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.validation import (
    ValidationError,
    validate_public_evoo_sunflower,
    validate_spectrum_set,
)


def _make_fs(values: np.ndarray, wavenumbers: np.ndarray) -> FoodSpectrumSet:
    meta = pd.DataFrame({"label": ["a"] * values.shape[0]})
    return FoodSpectrumSet(x=values, wavenumbers=wavenumbers, metadata=meta, modality="raman")


def test_validate_spectrum_set_ok() -> None:
    fs = _make_fs(np.ones((3, 4)), np.array([1.0, 2.0, 3.0, 4.0]))
    validate_spectrum_set(fs)


def test_validate_spectrum_set_non_monotonic_raises() -> None:
    fs = _make_fs(np.ones((2, 3)), np.array([1.0, 0.5, 2.0]))
    with pytest.raises(ValidationError):
        validate_spectrum_set(fs)


def test_validate_spectrum_set_nan_rejected() -> None:
    arr = np.ones((2, 3))
    arr[0, 0] = np.nan
    fs = _make_fs(arr, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValidationError):
        validate_spectrum_set(fs, allow_nan=False)


def test_validate_public_evoo_sunflower_fraction_ranges() -> None:
    wavenumbers = np.array([1.0, 2.0, 3.0])
    X = np.ones((3, 3))
    meta = pd.DataFrame({"mixture_fraction_evoo": [0.2, 0.5, 0.8]})
    fs = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=meta, modality="raman")
    validate_public_evoo_sunflower(fs)


def test_validate_public_evoo_sunflower_percent_warns() -> None:
    wavenumbers = np.array([1.0, 2.0, 3.0])
    X = np.ones((2, 3))
    meta = pd.DataFrame({"mixture_fraction_evoo": [20.0, 40.0]})
    fs = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=meta, modality="raman")
    with pytest.warns(RuntimeWarning):
        validate_public_evoo_sunflower(fs)


def test_validate_public_evoo_sunflower_out_of_range() -> None:
    wavenumbers = np.array([1.0, 2.0, 3.0])
    X = np.ones((1, 3))
    meta = pd.DataFrame({"mixture_fraction_evoo": [150.0]})
    fs = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=meta, modality="raman")
    with pytest.raises(ValidationError):
        validate_public_evoo_sunflower(fs)
