import numpy as np
import pandas as pd
import pytest

from foodspec.features.bands import integrate_bands
from foodspec.features.peaks import detect_peaks
from foodspec.features.ratios import compute_ratios


def test_integrate_bands_out_of_range_and_empty():
    X = np.ones((2, 5))
    wn = np.linspace(1000, 1010, 5)
    bands = [("band1", 2000, 2010)]
    df = integrate_bands(X, wn, bands)
    assert df.shape == (2, 1)
    assert df["band1"].isna().all()
    with pytest.raises(ValueError):
        integrate_bands(np.ones((2, 3, 1)), wn, [("a", 1, 2)])


def test_detect_peaks_prominence_width():
    wn = np.linspace(0, 10, 50)
    x = np.exp(-0.5 * (wn - 5) ** 2 / 0.5**2)
    df = detect_peaks(x, wn, prominence=0.1, width=1.0)
    assert not df.empty
    assert "peak_wavenumber" in df.columns


def test_compute_ratios_div_zero_and_missing():
    df = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 0.0]})
    ratios = compute_ratios(df, {"r": ("a", "b")})
    assert np.isnan(ratios.loc[0, "r"])
    with pytest.raises(ValueError):
        compute_ratios(df, {"bad": ("missing", "a")})
