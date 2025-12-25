import numpy as np
import pandas as pd

from foodspec.core.spectral_dataset import (
    HyperspectralDataset,
    PreprocessingConfig,
    SpectralDataset,
    harmonize_datasets,
)


def test_spectral_dataset_hdf5_roundtrip(tmp_path):
    wn = np.linspace(400.0, 1800.0, 128)
    X = np.random.RandomState(5).randn(10, wn.size)
    meta = pd.DataFrame({"sample_id": [f"S{i}" for i in range(10)], "group": ["A"] * 5 + ["B"] * 5})
    inst = {"protocol_name": "demo", "protocol_version": "1.0"}

    ds = SpectralDataset(wn, X, meta, inst)
    out = tmp_path / "spec.h5"
    ds.save_hdf5(out)
    ds2 = SpectralDataset.from_hdf5(out)

    assert ds2.spectra.shape == X.shape
    assert np.allclose(ds2.wavenumbers, wn)
    assert list(ds2.metadata.columns) == list(meta.columns)
    assert ds2.instrument_meta.get("protocol_name") == "demo"


def test_hyperspectral_hdf5_roundtrip(tmp_path):
    wn = np.linspace(600.0, 1600.0, 64)
    cube = np.random.RandomState(6).randn(8, 6, wn.size)
    meta = pd.DataFrame({"scan": ["A"]})

    h = HyperspectralDataset.from_cube(cube, wn, meta, {"protocol_name": "hs_demo"})
    out = tmp_path / "hs.h5"
    h.save_hdf5(out)
    h2 = HyperspectralDataset.from_hdf5(out)

    assert h2.shape_xy == (8, 6)
    assert np.allclose(h2.wavenumbers, wn)
    assert h2.spectra.shape == (8 * 6, wn.size)
    assert h2.instrument_meta.get("protocol_name") == "hs_demo"


def test_preprocess_pipeline_runs_and_logs():
    wn = np.linspace(500.0, 1500.0, 100)
    X = np.random.RandomState(7).randn(5, wn.size)
    ds = SpectralDataset(wn, X, pd.DataFrame({"id": range(5)}), {})
    opts = PreprocessingConfig(baseline_method="polynomial", baseline_order=2, smoothing_method="savgol", normalization="vector")
    ds2 = ds.preprocess(opts)
    assert ds2.spectra.shape == X.shape
    assert any("preprocess:" in log for log in ds2.logs)


def test_harmonize_datasets_aligns_to_longest_grid():
    wn1 = np.linspace(500.0, 1500.0, 80)
    wn2 = np.linspace(500.0, 1500.0, 100)
    X1 = np.random.RandomState(8).randn(3, wn1.size)
    X2 = np.random.RandomState(9).randn(4, wn2.size)
    d1 = SpectralDataset(wn1, X1, pd.DataFrame({"id": range(3)}), {})
    d2 = SpectralDataset(wn2, X2, pd.DataFrame({"id": range(4)}), {})

    out = harmonize_datasets([d1, d2])
    assert len(out) == 2
    # Both datasets should share the longest grid
    assert np.array_equal(out[0].wavenumbers, wn2)
    assert np.array_equal(out[1].wavenumbers, wn2)
