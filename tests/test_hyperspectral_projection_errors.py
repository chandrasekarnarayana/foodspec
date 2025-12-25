import numpy as np
import pandas as pd
import pytest

from foodspec.spectral_dataset import HyperspectralDataset


def test_hyperspectral_projection_invalid_band_raises(tmp_path):
    wn = np.linspace(600.0, 1600.0, 64)
    cube = np.random.RandomState(12).randn(6, 7, wn.size)
    meta = pd.DataFrame({"scan": ["E1"]})

    h = HyperspectralDataset.from_cube(cube, wn, meta, {"protocol_name": "proj_demo"})

    # band outside available wavenumbers -> expect all-NaN projection but no crash
    proj_oob = h.projection(mode="band", band=(2000.0, 2100.0))
    assert proj_oob.shape == h.shape_xy
    assert np.isnan(proj_oob).all()

    # malformed band tuple
    # malformed band tuple should error
    with pytest.raises(Exception):
        h.projection(mode="band", band=(900.0,))

    # inverted band range
    proj_inverted = h.projection(mode="band", band=(900.0, 800.0))
    assert proj_inverted.shape == h.shape_xy
