from __future__ import annotations

import numpy as np
import pandas as pd

from foodspec.core.api import FoodSpec


def test_foodspec_init_from_arrays(tmp_path):
    wavenumbers = np.linspace(400, 1800, 5)
    spectra = np.random.default_rng(0).normal(size=(3, 5))
    metadata = pd.DataFrame({"label": ["a", "b", "a"]})

    fs = FoodSpec(
        spectra,
        wavenumbers=wavenumbers,
        metadata=metadata,
        modality="raman",
        output_dir=tmp_path,
    )

    assert fs.data.x.shape == (3, 5)
    assert fs.output_dir.exists()
    summary = fs.summary()
    assert "FoodSpec Workflow Summary" in summary
