from __future__ import annotations

import numpy as np
import pandas as pd

from foodspec.data_objects.spectral_dataset import HyperspectralDataset


def test_mcr_als_unmixing_shapes():
    cube = np.random.RandomState(0).randn(2, 2, 8)
    wavenumbers = np.linspace(1000, 1100, 8)
    ds = HyperspectralDataset.from_cube(cube, wavenumbers, metadata=pd.DataFrame(index=range(4)))
    ds.shape_xy = (2, 2)
    result = ds.unmix_mcr_als(n_components=2, max_iter=10)
    assert result["abundances"].shape == (2, 2, 2)
    assert result["components"].shape == (2, 8)
