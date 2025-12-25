"""Comprehensive tests for preprocessing modules to achieve 90%+ coverage."""

import numpy as np

from foodspec.preprocess.baseline import ALSBaseline, PolynomialBaseline, RubberbandBaseline
from foodspec.preprocess.normalization import (
    AreaNormalizer,
    InternalPeakNormalizer,
    MSCNormalizer,
    SNVNormalizer,
    VectorNormalizer,
)
from foodspec.preprocess.smoothing import MovingAverageSmoother, SavitzkyGolaySmoother
from foodspec.preprocess.spikes import correct_cosmic_rays


def test_als_baseline():
    rng = np.random.RandomState(42)
    X = rng.randn(5, 100) + np.linspace(0, 2, 100)

    als = ALSBaseline(lambda_=1e4, p=0.01, max_iter=15)
    als.fit(X)
    corrected = als.transform(X)

    assert corrected.shape == X.shape
    assert isinstance(corrected, np.ndarray)


def test_rubberband_baseline():
    rng = np.random.RandomState(43)
    X = rng.randn(3, 150)

    rb = RubberbandBaseline()
    rb.fit(X)
    corrected = rb.transform(X)

    assert corrected.shape == X.shape


def test_polynomial_baseline():
    rng = np.random.RandomState(44)
    X = rng.randn(3, 80)

    for degree in [2, 3, 5]:
        poly = PolynomialBaseline(degree=degree)
        poly.fit(X)
        corrected = poly.transform(X)
        assert corrected.shape == X.shape


def test_savgol_smoother():
    rng = np.random.RandomState(45)
    X = rng.randn(4, 100)

    sg = SavitzkyGolaySmoother(window_length=11, polyorder=2)
    sg.fit(X)
    smoothed = sg.transform(X)

    assert smoothed.shape == X.shape


def test_moving_average():
    rng = np.random.RandomState(47)
    X = rng.randn(3, 100)

    for window_size in [3, 5, 7]:
        ma = MovingAverageSmoother(window_size=window_size)
        ma.fit(X)
        smoothed = ma.transform(X)
        assert smoothed.shape == X.shape


def test_vector_normalizer():
    rng = np.random.RandomState(49)
    X = rng.randn(5, 100) * 10 + 50

    for norm_type in ["l1", "l2", "max"]:
        vn = VectorNormalizer(norm=norm_type)
        vn.fit(X)
        normalized = vn.transform(X)
        assert normalized.shape == X.shape

        if norm_type == "l2":
            norms = np.linalg.norm(normalized, axis=1)
            assert np.allclose(norms, 1.0)


def test_area_normalizer():
    rng = np.random.RandomState(52)
    X = np.abs(rng.randn(3, 100)) * 10

    an = AreaNormalizer()
    an.fit(X)
    normalized = an.transform(X)

    assert normalized.shape == X.shape


def test_internal_peak_normalizer():
    wavenumbers = np.linspace(500, 1500, 100)
    X = np.random.RandomState(53).randn(3, 100)
    X[:, 50] += 10  # Add peak in middle

    pn = InternalPeakNormalizer(target_wavenumber=1000.0, window=50.0)
    pn.fit(X, wavenumbers=wavenumbers)
    normalized = pn.transform(X, wavenumbers=wavenumbers)

    assert normalized.shape == X.shape


def test_snv_normalizer():
    rng = np.random.RandomState(54)
    X = rng.randn(4, 90) * 5 + 20

    snv = SNVNormalizer()
    snv.fit(X)
    normalized = snv.transform(X)

    assert normalized.shape == X.shape
    means = np.mean(normalized, axis=1)
    stds = np.std(normalized, axis=1)
    assert np.allclose(means, 0.0, atol=1e-10)
    assert np.allclose(stds, 1.0, atol=1e-6)


def test_msc_normalizer():
    rng = np.random.RandomState(55)
    reference = rng.randn(100)
    X = np.vstack([reference * 1.2 + 0.5, reference * 0.8 + 0.3, reference * 1.5 + 0.1])

    msc = MSCNormalizer()
    msc.fit(X)
    normalized = msc.transform(X)

    assert normalized.shape == X.shape


def test_correct_cosmic_rays():
    rng = np.random.RandomState(56)
    X = rng.randn(3, 100)
    X[0, 20] += 15  # Add spikes
    X[1, 50] += 20

    corrected, report = correct_cosmic_rays(X, window=5, zscore_thresh=5.0)

    assert corrected.shape == X.shape
    assert hasattr(report, "total_spikes")
    assert report.total_spikes > 0


def test_preprocessing_edge_cases():
    """Test edge cases across preprocessing modules."""
    # Single spectrum
    X_single = np.random.RandomState(60).randn(1, 50)

    als = ALSBaseline()
    als.fit(X_single)
    result = als.transform(X_single)
    assert result.shape == X_single.shape

    # Near-zero handling
    X_zero = np.zeros((2, 50))
    X_zero[1, :] = 1.0

    snv = SNVNormalizer()
    snv.fit(X_zero)
    normalized = snv.transform(X_zero)
    assert normalized.shape == X_zero.shape
    assert np.all(np.isfinite(normalized))
