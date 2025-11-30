import numpy as np

from foodspec.preprocess.raman import CosmicRayRemover


def test_cosmic_ray_remover_despikes():
    wn = np.linspace(800, 1200, 200)
    base = np.sin(wn / 50) * 0.1 + 1.0
    spikes = base.copy()
    spike_indices = [50, 120, 150]
    spikes[spike_indices] += 5.0
    X = np.vstack([spikes, spikes * 1.01])

    remover = CosmicRayRemover(window=7, sigma_thresh=6.0)
    X_clean = remover.transform(X)

    assert X_clean.shape == X.shape
    assert np.isfinite(X_clean).all()
    for idx in spike_indices:
        assert X_clean[0, idx] < spikes[idx] / 2.0

