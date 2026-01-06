"""Synthetic heating quality dataset for demos and tests."""

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet


def synthetic_heating_dataset():
    """Generate a deterministic synthetic Raman spectrum dataset for heating quality demos.
    
    Creates 20 synthetic spectra over a heating time course (0-30 time units) with two
    peaks that change in intensity to simulate a heating experiment. Fixed random seed
    ensures reproducibility.
    
    Returns
    -------
    FoodSpectrumSet
        Synthetic dataset with:
        - 20 spectra (n_samples=20)
        - 200 wavenumbers (600-1800 cm^-1)
        - metadata including sample_id and heating_time
        - modality: 'raman'
    """
    rng = np.random.default_rng(0)
    wavenumbers = np.linspace(600, 1800, 200)
    n = 20
    # synthetic baseline + two peaks changing with time
    times = np.linspace(0, 30, n)
    spectra = []
    for t in times:
        peak1 = np.exp(-0.5 * ((wavenumbers - 1655) / 8) ** 2) * (1.0 - 0.01 * t)
        peak2 = np.exp(-0.5 * ((wavenumbers - 1742) / 8) ** 2) * (1.0 + 0.005 * t)
        baseline = 0.02 + 0.00001 * (wavenumbers - 1200) ** 2
        noise = rng.normal(0, 0.01, size=wavenumbers.shape)
        spectra.append(peak1 + peak2 + baseline + noise)
    x = np.vstack(spectra)
    metadata = pd.DataFrame({"sample_id": [f"s{i:03d}" for i in range(n)], "heating_time": times})
    return FoodSpectrumSet(x=x, wavenumbers=wavenumbers, metadata=metadata, modality="raman")
