import numpy as np
import h5py

from foodspec.spectral_dataset import SpectralDataset


def test_spectral_dataset_legacy_minimal_hdf5_loads(tmp_path):
    wn = np.linspace(500.0, 1500.0, 32)
    X = np.random.RandomState(11).randn(4, wn.size)

    path = tmp_path / "legacy_minimal.h5"
    with h5py.File(path, "w") as f:
        # Legacy layout expects root-level datasets with *_legacy names when modern group is absent
        f.create_dataset("spectra_legacy", data=X)
        f.create_dataset("wavenumbers_legacy", data=wn)
        # Minimal metadata and instrument meta to satisfy reader
        meta_json = "{}"  # empty DataFrame JSON would be complex; allow empty object
        f.create_dataset("metadata_json", data=np.bytes_(meta_json))
        inst_json = "{}"
        f.create_dataset("instrument_meta", data=np.bytes_(inst_json))
        f.create_dataset("logs", data=np.bytes_(""))
        # Optional history
        f.create_dataset("history_json", data=np.bytes_("[]"))

    ds = SpectralDataset.from_hdf5(path)
    assert ds.spectra.shape == X.shape
    assert np.allclose(ds.wavenumbers, wn)
    # metadata may be minimal/empty, but object should be usable
    assert hasattr(ds, "metadata")
    assert ds.instrument_meta is not None
