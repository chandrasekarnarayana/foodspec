import numpy as np
import pytest

from foodspec.preprocess.cropping import RangeCropper
from foodspec.preprocess.normalization import VectorNormalizer


def test_range_cropper_empty_mask_raises():
    cropper = RangeCropper(min_wn=10, max_wn=20)
    wavenumbers = np.array([1.0, 2.0, 3.0])
    X = np.ones((2, 3))
    with pytest.raises(ValueError):
        cropper.transform(X, wavenumbers)


def test_vector_normalizer_zero_norm():
    norm = VectorNormalizer(norm="l2")
    X = np.array([[0.0, 0.0]])
    out = norm.transform(X)
    assert np.all(out == 0)
