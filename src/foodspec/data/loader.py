"""Example data loaders."""
from __future__ import annotations


from typing import Optional

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet

__all__ = ["load_example_oils"]


def load_example_oils(seed: Optional[int] = 0) -> FoodSpectrumSet:
    """Load a small synthetic edible oils dataset.

    Notes
    -----
    This currently generates synthetic data; in future it can be replaced with
    reads from packaged resources via ``importlib.resources.files``.
    """

    rng = np.random.default_rng(seed)
    wavenumbers = np.linspace(800, 1800, 250)
    n_per_class = 15
    classes = ["olive", "sunflower"]
    spectra = []
    labels = []
    for cls in classes:
        for _ in range(n_per_class):
            base = np.zeros_like(wavenumbers)
            if cls == "olive":
                base += 1.2 * np.exp(-0.5 * ((wavenumbers - 1655) / 12) ** 2)
                base += 1.0 * np.exp(-0.5 * ((wavenumbers - 1742) / 14) ** 2)
            else:
                base += 0.8 * np.exp(-0.5 * ((wavenumbers - 1655) / 12) ** 2)
                base += 1.3 * np.exp(-0.5 * ((wavenumbers - 1742) / 14) ** 2)
            noise = rng.normal(0, 0.02, size=wavenumbers.shape)
            spectra.append(base + noise)
            labels.append(cls)

    X = np.vstack(spectra)
    metadata = pd.DataFrame(
        {
            "sample_id": [f"{lbl}_{i}" for i, lbl in enumerate(labels)],
            "oil_type": labels,
            "matrix": ["pure-oil"] * len(labels),
        }
    )
    return FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=metadata, modality="raman")
