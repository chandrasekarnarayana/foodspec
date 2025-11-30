import numpy as np

from foodspec.preprocess.baseline import ALSBaseline, RubberbandBaseline


def _synthetic_spectra(n_samples: int = 3, n_points: int = 500):
    rng = np.random.default_rng(0)
    wn = np.linspace(500, 2000, n_points)
    baseline = 0.0001 * (wn - wn.min()) ** 2 + 0.05 * (wn - wn.min())
    centers = [800, 1200, 1700]
    heights = [1.0, 0.8, 1.2]
    peaks = np.zeros((n_samples, n_points))
    for i in range(n_samples):
        spec = baseline.copy()
        for c, h in zip(centers, heights):
            spec += h * np.exp(-0.5 * ((wn - c) / 20.0) ** 2)
        noise = rng.normal(0, 0.02, size=n_points)
        peaks[i] = spec + noise
    true_peaks = heights
    return peaks, wn, baseline, centers, true_peaks


def _non_peak_mask(wn, centers, window=40):
    mask = np.ones_like(wn, dtype=bool)
    for c in centers:
        mask &= (wn < c - window) | (wn > c + window)
    return mask


def test_baseline_corrections_improve_background():
    X, wn, baseline, centers, true_heights = _synthetic_spectra()
    non_peak_mask = _non_peak_mask(wn, centers)

    pre_mean = X[:, non_peak_mask].mean()

    als = ALSBaseline(lambda_=1e5, p=0.01, max_iter=20)
    X_als = als.transform(X)
    post_mean_als = X_als[:, non_peak_mask].mean()
    assert abs(post_mean_als) < abs(pre_mean) / 2.0

    rb = RubberbandBaseline()
    X_rb = rb.transform(X)
    post_mean_rb = X_rb[:, non_peak_mask].mean()
    assert abs(post_mean_rb) < abs(pre_mean) / 2.0

    # Peak preservation (approximate)
    for corrected in (X_als, X_rb):
        for c, h_true in zip(centers, true_heights):
            window = (wn >= c - 5) & (wn <= c + 5)
            peak_height = corrected[:, window].max(axis=1).mean()
            assert np.isclose(peak_height, h_true, rtol=0.3)
