"""
Quickstart script for mixture analysis (NNLS) using foodspec.
Run with: python examples/mixture_analysis_quickstart.py
"""

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.chemometrics.mixture import run_mixture_analysis_workflow


def _synthetic_mixtures():
    rng = np.random.default_rng(0)
    wavenumbers = np.linspace(600, 1800, 120)
    # Two pure components
    s1 = np.exp(-0.5 * ((wavenumbers - 1000) / 12) ** 2)
    s2 = np.exp(-0.5 * ((wavenumbers - 1400) / 10) ** 2)

    coeffs = []
    mixtures = []
    for frac in np.linspace(0, 1, 6):
        c1 = frac
        c2 = 1 - frac
        spectrum = c1 * s1 + c2 * s2 + rng.normal(0, 0.01, size=wavenumbers.shape)
        mixtures.append(spectrum)
        coeffs.append((c1, c2))
    mixtures = np.vstack(mixtures)
    pure = np.vstack([s1, s2])
    metadata = pd.DataFrame({"sample_id": [f"m{i:02d}" for i in range(len(mixtures))]})
    return (
        FoodSpectrumSet(x=mixtures, wavenumbers=wavenumbers, metadata=metadata, modality="raman"),
        FoodSpectrumSet(x=pure, wavenumbers=wavenumbers, metadata=pd.DataFrame({"sample_id": ["pure1", "pure2"]}), modality="raman"),
        np.array(coeffs),
    )


def main():
    mix, pure, true_coeffs = _synthetic_mixtures()
    res = run_mixture_analysis_workflow(mixtures=mix.x, pure_spectra=pure.x, mode="nnls")
    print("Estimated coefficients:\n", res["coefficients"])
    print("Residual norms:", res["residual_norms"])


if __name__ == "__main__":
    main()
