import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.cropping import RangeCropper, crop_spectrum_set
from foodspec.preprocess.normalization import (
    AreaNormalizer,
    InternalPeakNormalizer,
    VectorNormalizer,
)
from foodspec.preprocess.smoothing import MovingAverageSmoother, SavitzkyGolaySmoother


def _synthetic_spectra(n_samples: int = 3, n_points: int = 200):
    rng = np.random.default_rng(0)
    wavenumbers = np.linspace(500, 1800, n_points)
    baseline = 0.001 * (wavenumbers - wavenumbers.min()) ** 2 + 0.1
    peak_center = 1100
    peak = np.exp(-0.5 * ((wavenumbers - peak_center) / 20.0) ** 2)
    spectra = []
    for i in range(n_samples):
        noise = rng.normal(0, 0.05, size=n_points)
        spectra.append(baseline + 2 * peak + noise)
    X = np.vstack(spectra)
    metadata = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n_samples)]})
    return X, wavenumbers, metadata


def test_baseline_als_removes_polynomial_offset():
    X, wavenumbers, metadata = _synthetic_spectra()
    als = ALSBaseline(lambda_=1e5, p=0.01, max_iter=15)
    corrected = als.transform(X)
    # Baseline near spectrum edges should be close to zero after correction.
    edge_mean = corrected[:, :20].mean()
    assert abs(edge_mean) < 0.2


def test_smoothing_reduces_noise_variance():
    X, wavenumbers, _ = _synthetic_spectra()
    # Create clean signal without noise to estimate residual variance
    clean_baseline = 0.001 * (wavenumbers - wavenumbers.min()) ** 2 + 0.1
    clean_peak = np.exp(-0.5 * ((wavenumbers - 1100) / 20.0) ** 2)
    clean_signal = clean_baseline + 2 * clean_peak
    smoother = SavitzkyGolaySmoother(window_length=9, polyorder=3)
    smoothed = smoother.transform(X)
    residual_raw = X - clean_signal
    residual_smoothed = smoothed - clean_signal
    assert residual_smoothed.var() < residual_raw.var()

    ma = MovingAverageSmoother(window_size=5)
    smoothed_ma = ma.transform(X)
    assert smoothed_ma.shape == X.shape


def test_normalization_methods():
    X, wavenumbers, _ = _synthetic_spectra()
    vn = VectorNormalizer(norm="l2")
    X_norm = vn.transform(X)
    norms = np.linalg.norm(X_norm, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    area_norm = AreaNormalizer()
    X_area = area_norm.transform(X)
    areas = np.trapezoid(X_area, axis=1)
    assert np.allclose(areas, 1.0, atol=1e-3)

    ipn = InternalPeakNormalizer(target_wavenumber=1100, window=20)
    X_ipn = ipn.transform(X, wavenumbers=wavenumbers)
    mask = (wavenumbers >= 1090) & (wavenumbers <= 1110)
    ref_mean = X_ipn[:, mask].mean(axis=1)
    assert np.allclose(ref_mean, 1.0, atol=1e-2)


def test_cropping_reduces_wavenumber_range():
    X, wavenumbers, metadata = _synthetic_spectra()
    cropper = RangeCropper(min_wn=800, max_wn=1200)
    X_cropped, wn_cropped = cropper.transform(X, wavenumbers)
    assert wn_cropped.min() >= 800 and wn_cropped.max() <= 1200
    assert X_cropped.shape[1] < X.shape[1]

    ds = FoodSpectrumSet(
        x=X,
        wavenumbers=wavenumbers,
        metadata=metadata,
        modality="raman",
    )
    ds_cropped = crop_spectrum_set(ds, min_wn=800, max_wn=1200)
    assert ds_cropped.x.shape == X_cropped.shape
    assert np.allclose(ds_cropped.wavenumbers, wn_cropped)
