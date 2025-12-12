from pathlib import Path

import numpy as np
import pandas as pd

from foodspec.spectral_dataset import SpectralDataset


def test_hdf5_schema_roundtrip(tmp_path: Path):
    wn = np.array([1000.0, 1010.0])
    spectra = np.array([[1.0, 2.0], [3.0, 4.0]])
    meta = pd.DataFrame({"oil_type": ["A", "B"]})
    ds = SpectralDataset(wn, spectra, meta, instrument_meta={"instrument_id": "test"})
    path = tmp_path / "spec.h5"
    ds.save_hdf5(path)
    ds2 = SpectralDataset.from_hdf5(path)
    assert np.allclose(ds2.wavenumbers, wn)
    assert ds2.metadata.shape[0] == meta.shape[0]
