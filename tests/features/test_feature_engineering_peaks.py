import numpy as np

from foodspec.features.peaks import extract_peak_features
from foodspec.features.schema import PeakSpec


def test_extract_peak_features_with_linear_baseline():
    wn = np.linspace(1000, 1100, 101)
    peak = np.exp(-0.5 * ((wn - 1050.0) / 2.0) ** 2)
    baseline = 0.01 * (wn - 1000.0)
    X = np.vstack([peak + baseline, peak + baseline * 0.5])

    spec = PeakSpec(name="p1050", center=1050.0, window=5.0, baseline="linear")
    df, info = extract_peak_features(X, wn, [spec], metrics=("height", "area"))

    assert "p1050" in df.columns
    assert df.shape[0] == 2
    assert np.isclose(df["p1050"].iloc[0], 1.0, atol=0.2)
    assert df["p1050_area"].iloc[0] > 0
    assert info
