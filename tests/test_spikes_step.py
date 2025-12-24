import numpy as np
from foodspec.preprocess.spikes import correct_cosmic_rays


def test_correct_cosmic_rays_counts():
    # Create a matrix with intentional spikes
    X = np.vstack([
        np.array([1, 1, 100, 1, 1], dtype=float),
        np.array([2, 2, 2, 2, 2], dtype=float),
    ])
    Xc, report = correct_cosmic_rays(X, window=3, zscore_thresh=5.0)
    assert report.total_spikes >= 1
    assert report.spikes_per_spectrum[0] >= 1
    # Corrected spike should be near neighbors
    assert Xc[0, 2] < 100
