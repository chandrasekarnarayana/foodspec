import numpy as np
import pandas as pd
import pytest

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.validation import ValidationError, validate_spectrum_set


def test_validate_allow_nan_and_non_monotonic():
    X = np.array([[1.0, 2.0, 3.0]])
    wn = np.array([1.0, 2.0, 3.0])
    meta = pd.DataFrame({"sample_id": ["a"]})
    fs = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")
    # modify to non-monotonic and inject NaN after construction
    fs.wavenumbers = np.array([2.0, 1.0, 3.0])
    fs.x[0, 1] = np.nan
    with pytest.raises(ValidationError):
        validate_spectrum_set(fs)
    validate_spectrum_set(fs, allow_nan=True, check_monotonic=False)


def test_validate_metadata_length_mismatch():
    X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    wn = np.array([1.0, 2.0, 3.0])
    meta = pd.DataFrame({"sample_id": ["a"]})
    with pytest.raises(ValueError):
        FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")


def test_validate_public_evoo_fraction():
    X = np.ones((3, 3))
    wn = np.array([1.0, 2.0, 3.0])
    meta = pd.DataFrame({"mixture_fraction_evoo": [0, 50, 120]})
    fs = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")
    # values >100 should raise
    with pytest.raises(ValidationError):
        from foodspec.validation import validate_public_evoo_sunflower

        validate_public_evoo_sunflower(fs)
