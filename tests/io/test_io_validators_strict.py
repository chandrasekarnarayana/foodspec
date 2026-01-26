import numpy as np
import pytest

from foodspec.core.errors import FoodSpecValidationError
from foodspec.io.validators import (
    validate_intensity_array,
    validate_metadata,
    validate_wavenumber_axis,
)


def test_validate_wavenumber_axis_non_monotonic():
    axis = np.array([1000.0, 1001.0, 1000.5])
    with pytest.raises(FoodSpecValidationError):
        validate_wavenumber_axis(axis)


def test_validate_intensity_array_nan():
    intensities = np.array([1.0, np.nan])
    with pytest.raises(FoodSpecValidationError):
        validate_intensity_array(intensities)


def test_validate_metadata_missing_required():
    with pytest.raises(FoodSpecValidationError):
        validate_metadata({}, required={"sample_id"}, optional=set())


def test_validate_metadata_bad_unit():
    with pytest.raises(FoodSpecValidationError):
        validate_metadata({"unit": "banana"}, required=set(), optional={"unit"})
