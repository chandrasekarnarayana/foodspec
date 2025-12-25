import numpy as np
import pandas as pd
import pytest

from foodspec.features.bands import integrate_bands


def test_integrate_bands_linear_ramp():
    wn = np.array([0.0, 1.0, 2.0, 3.0])
    X = np.vstack([wn, wn + 1])  # two samples
    bands = [("band1", 0.5, 2.5)]
    df = integrate_bands(X, wn, bands)
    # Trapezoid integration of y=x over [0.5, 2.5] gives 1.5; y=x+1 gives 2.5
    expected = pd.Series([1.5, 2.5], name="band1")
    assert np.allclose(df["band1"].to_numpy(), expected.to_numpy(), atol=1e-6)


def test_integrate_bands_invalid_range_and_empty_mask():
    wn = np.array([0.0, 1.0])
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        integrate_bands(X, wn, [("bad", 2.0, 1.0)])
    df = integrate_bands(X, wn, [("empty", 5.0, 6.0)])
    assert np.isnan(df["empty"]).all()
