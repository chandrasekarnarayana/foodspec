import numpy as np

from foodspec.features.bands import integrate_bands


def test_integrate_bands_known_area():
    wn = np.linspace(0, 10, 101)
    X = np.vstack([wn, wn * 2])  # linear ramps
    bands = [("low", 0, 5), ("high", 5, 10)]
    df = integrate_bands(X, wn, bands)
    # area of linear ramp y=x over [0,5] ~= 0.5*5^2 = 12.5
    assert np.allclose(df["low"].iloc[0], 12.5, atol=0.1)
    # second spectrum is doubled
    assert np.allclose(df["low"].iloc[1], 25.0, atol=0.1)
