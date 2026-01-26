import numpy as np

from foodspec.features.bands import extract_band_features
from foodspec.features.schema import BandSpec


def test_extract_band_features_deterministic():
    wn = np.linspace(1000, 1100, 101)
    X = np.vstack([np.sin(wn / 10.0), np.cos(wn / 10.0)])
    bands = [BandSpec(name="band_a", start=1020.0, end=1050.0, baseline="min")]

    df1, info1 = extract_band_features(X, wn, bands, metrics=("integral", "mean"))
    df2, info2 = extract_band_features(X, wn, bands, metrics=("integral", "mean"))

    assert df1.equals(df2)
    assert info1 and info2
