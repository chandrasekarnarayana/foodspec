from __future__ import annotations

import numpy as np

from foodspec.simulation.spectra_sim import NoiseModel, SpectraSimulator


def test_simulator_generates_mixture_dataset():
    sim = SpectraSimulator(n_wavelengths=50, random_state=0)
    sim.add_noise_model(NoiseModel("gaussian", std=0.01))
    X, y, meta = sim.generate_mixture_dataset(n_samples=5, n_components=3)

    assert X.shape == (5, 50)
    assert y.shape == (5, 3)
    assert "pure_spectra" in meta
    assert np.isfinite(X).all()
