import numpy as np
import pandas as pd

from foodspec.features.bands import integrate_bands
from foodspec.features.fingerprint import (
    correlation_similarity_matrix,
    cosine_similarity_matrix,
)
from foodspec.features.peaks import PeakFeatureExtractor, detect_peaks
from foodspec.features.ratios import RatioFeatureGenerator, compute_ratios


def _synthetic_gaussians(n_samples: int = 4, noise: float = 0.01):
    rng = np.random.default_rng(0)
    wavenumbers = np.linspace(800, 1800, 500)
    centers = [1000, 1500]
    widths = [20, 30]
    spectra = []
    for i in range(n_samples):
        y = np.zeros_like(wavenumbers)
        for c, w in zip(centers, widths):
            y += np.exp(-0.5 * ((wavenumbers - c) / w) ** 2)
        y += rng.normal(0, noise, size=wavenumbers.shape)
        spectra.append(y)
    X = np.vstack(spectra)
    return X, wavenumbers


def test_peak_feature_extractor_recovers_peaks():
    X, wavenumbers = _synthetic_gaussians()
    extractor = PeakFeatureExtractor(expected_peaks=[1000, 1500], tolerance=10.0)
    feats = extractor.transform(X, wavenumbers=wavenumbers)
    names = extractor.get_feature_names_out()
    assert feats.shape == (X.shape[0], len(names))
    # heights should be close to 1.0 for synthetic peaks
    heights = feats[:, [0, 2]]  # height features if ordering height, area
    assert np.allclose(heights, 1.0, atol=0.2)
    # areas positive
    assert np.all(feats[:, 1::2] > 0)


def test_detect_peaks_outputs_dataframe():
    X, wavenumbers = _synthetic_gaussians(n_samples=1)
    df = detect_peaks(X[0], wavenumbers, prominence=0.1)
    assert {"peak_index", "peak_wavenumber", "peak_intensity"}.issubset(df.columns)
    assert len(df) >= 2


def test_integrate_bands_matches_numeric():
    X, wavenumbers = _synthetic_gaussians()
    bands = [("band1", 950, 1050), ("band2", 1450, 1550)]
    df = integrate_bands(X, wavenumbers, bands=bands)
    assert list(df.columns) == ["band1", "band2"]
    assert df.shape == (X.shape[0], 2)
    assert np.all(df["band1"] > 0)
    assert np.all(df["band2"] > 0)


def test_compute_ratios_and_similarity():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 0.0]})
    ratios = compute_ratios(df, {"r1": ("a", "b")})
    assert np.isnan(ratios.loc[1, "r1"])  # division by zero handled

    gen = RatioFeatureGenerator({"r2": ("a", "a")})
    ratios2 = gen.transform(df)
    assert np.allclose(ratios2["r2"], 1.0)

    X_ref = np.array([[1, 0], [0, 1]], dtype=float)
    X_query = np.array([[1, 0], [1, 1]], dtype=float)
    cos = cosine_similarity_matrix(X_ref, X_query)
    corr = correlation_similarity_matrix(X_ref, X_query)
    assert cos.shape == (2, 2)
    assert corr.shape == (2, 2)
    assert np.isclose(cos[0, 0], 1.0)
    assert corr[0, 0] > 0.9

