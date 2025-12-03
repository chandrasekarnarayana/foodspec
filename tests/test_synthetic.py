import numpy as np

from foodspec.synthetic import (
    generate_synthetic_ftir_spectrum,
    generate_synthetic_raman_spectrum,
)


def test_generate_raman_shape_and_noise():
    wn, y = generate_synthetic_raman_spectrum(noise_level=0.0)
    assert wn.shape == y.shape
    assert wn.min() >= 350 and wn.max() <= 1850
    # Expect non-zero signal
    assert np.isfinite(y).all()
    assert y.max() > 0.1


def test_generate_ftir_shape_and_baseline():
    wn, y = generate_synthetic_ftir_spectrum(noise_level=0.0, baseline_amp=0.05)
    assert wn.shape == y.shape
    assert wn.min() >= 750 and wn.max() <= 3650
    assert np.isfinite(y).all()
    # Baseline should vary a bit
    assert y.max() - y.min() > 0.01
