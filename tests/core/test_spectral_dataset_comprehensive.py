"""Comprehensive tests for SpectralDataset to increase coverage."""

import numpy as np
import pandas as pd
import pytest

from foodspec.core.spectral_dataset import (
    HyperspectralDataset,
    PreprocessingConfig,
    SpectralDataset,
    baseline_als,
    baseline_polynomial,
    baseline_rubberband,
    normalize_matrix,
    remove_spikes,
    smooth_signal,
)
from foodspec.features.rq import PeakDefinition


def test_spectral_dataset_copy():
    """Test copy method creates independent copy."""
    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5)})

    ds = SpectralDataset(wn, spectra, meta, {"instrument": "raman"}, ["log1"], [{"step": 1}])
    ds_copy = ds.copy()

    # Modify original
    ds.spectra[0, 0] = 999
    ds.logs.append("new log")

    # Copy should be unchanged
    assert ds_copy.spectra[0, 0] != 999
    assert len(ds_copy.logs) == 1


def test_spectral_dataset_preprocess_baseline_methods():
    """Test preprocessing with different baseline methods."""
    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(3, 100) + 10  # Add baseline
    ds = SpectralDataset(wn, spectra)

    # Test ALS baseline
    config_als = PreprocessingConfig(baseline_method="als", baseline_enabled=True)
    ds_als = ds.preprocess(config_als)
    assert ds_als.spectra.shape == spectra.shape
    assert len(ds_als.logs) > 0

    # Test rubberband baseline
    config_rb = PreprocessingConfig(baseline_method="rubberband")
    ds_rb = ds.preprocess(config_rb)
    assert ds_rb.spectra.shape == spectra.shape

    # Test polynomial baseline
    config_poly = PreprocessingConfig(baseline_method="polynomial", baseline_order=3)
    ds_poly = ds.preprocess(config_poly)
    assert ds_poly.spectra.shape == spectra.shape


def test_spectral_dataset_preprocess_smoothing():
    """Test preprocessing with different smoothing methods."""
    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(3, 100)
    ds = SpectralDataset(wn, spectra)

    # Test savgol smoothing
    config_savgol = PreprocessingConfig(smoothing_method="savgol", smoothing_window=7)
    ds_smooth = ds.preprocess(config_savgol)
    assert ds_smooth.spectra.shape == spectra.shape

    # Test moving average
    config_ma = PreprocessingConfig(smoothing_method="moving_average", smoothing_window=5)
    ds_ma = ds.preprocess(config_ma)
    assert ds_ma.spectra.shape == spectra.shape


def test_spectral_dataset_preprocess_normalization():
    """Test preprocessing with different normalization modes."""
    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(3, 100) + 10
    ds = SpectralDataset(wn, spectra)

    # Test reference normalization
    config_ref = PreprocessingConfig(normalization="reference", reference_wavenumber=1000)
    ds_ref = ds.preprocess(config_ref)
    assert ds_ref.spectra.shape == spectra.shape

    # Test vector normalization
    config_vec = PreprocessingConfig(normalization="vector")
    ds_vec = ds.preprocess(config_vec)
    assert ds_vec.spectra.shape == spectra.shape

    # Test area normalization
    config_area = PreprocessingConfig(normalization="area")
    ds_area = ds.preprocess(config_area)
    assert ds_area.spectra.shape == spectra.shape

    # Test max normalization
    config_max = PreprocessingConfig(normalization="max")
    ds_max = ds.preprocess(config_max)
    assert ds_max.spectra.shape == spectra.shape


def test_spectral_dataset_spike_removal():
    """Test spike removal."""
    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(3, 100)
    spectra[0, 50] = 100  # Add a spike
    ds = SpectralDataset(wn, spectra)

    config = PreprocessingConfig(spike_removal=True, spike_zscore_thresh=5.0)
    ds_clean = ds.preprocess(config)

    assert ds_clean.spectra[0, 50] < 50  # Spike should be reduced


def test_spectral_dataset_to_peaks():
    """Test peak extraction."""
    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(5, 100) + 10
    meta = pd.DataFrame({"id": range(5)})
    ds = SpectralDataset(wn, spectra, meta)

    peaks = [
        PeakDefinition(name="peak1", column="I_1000", wavenumber=1000, window=(950, 1050), mode="max"),
        PeakDefinition(name="peak2", column="I_1200", wavenumber=1200, window=(1150, 1250), mode="area"),
    ]

    result = ds.to_peaks(peaks)
    assert "peak1" in result.columns
    assert "peak2" in result.columns
    assert len(result) == 5


def test_spectral_dataset_hdf5_roundtrip(tmp_path):
    """Test save and load HDF5."""
    pytest.importorskip("h5py")

    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5), "label": ["A", "B", "A", "B", "C"]})
    instrument = {"name": "raman_spectrometer", "protocol_name": "test"}
    logs = ["log1", "log2"]
    history = [{"step": "baseline", "params": {"method": "als"}}]

    ds = SpectralDataset(wn, spectra, meta, instrument, logs, history)

    path = tmp_path / "test.h5"
    ds.save_hdf5(path)

    ds_loaded = SpectralDataset.from_hdf5(path)

    assert np.allclose(ds_loaded.wavenumbers, wn)
    assert ds_loaded.spectra.shape == spectra.shape
    assert len(ds_loaded.metadata) == 5
    assert "raman_spectrometer" in str(ds_loaded.instrument_meta)


def test_baseline_als():
    """Test ALS baseline correction."""
    y = np.random.randn(100) + np.linspace(0, 10, 100)  # Signal with baseline
    baseline = baseline_als(y, lam=1e5, p=0.01, niter=5)
    assert len(baseline) == len(y)
    assert np.all(baseline < y + 1)


def test_baseline_rubberband():
    """Test rubberband baseline correction."""
    x = np.linspace(0, 100, 100)
    y = np.random.randn(100) + 10
    baseline = baseline_rubberband(x, y)
    assert len(baseline) == len(y)


def test_baseline_polynomial():
    """Test polynomial baseline correction."""
    x = np.linspace(0, 100, 100)
    y = x**2 + np.random.randn(100)
    baseline = baseline_polynomial(x, y, order=3)
    assert len(baseline) == len(y)


def test_smooth_signal_savgol():
    """Test Savitzky-Golay smoothing."""
    y = np.random.randn(100)
    smoothed = smooth_signal(y, method="savgol", window=7, polyorder=3)
    assert len(smoothed) == len(y)


def test_smooth_signal_moving_average():
    """Test moving average smoothing."""
    y = np.random.randn(100)
    smoothed = smooth_signal(y, method="moving_average", window=5)
    assert len(smoothed) == len(y)


def test_normalize_matrix_modes():
    """Test different normalization modes."""
    wn = np.linspace(500, 1500, 100)
    X = np.random.randn(5, 100) + 10

    # Test reference normalization
    X_ref = normalize_matrix(X, "reference", wn, 1000)
    assert X_ref.shape == X.shape

    # Test vector normalization
    X_vec = normalize_matrix(X, "vector", wn, 1000)
    assert X_vec.shape == X.shape

    # Test area normalization
    X_area = normalize_matrix(X, "area", wn, 1000)
    assert X_area.shape == X.shape

    # Test max normalization
    X_max = normalize_matrix(X, "max", wn, 1000)
    assert X_max.shape == X.shape

    # Test none normalization
    X_none = normalize_matrix(X, "none", wn, 1000)
    assert np.allclose(X_none, X)


def test_remove_spikes():
    """Test spike removal."""
    y = np.random.randn(100)
    y[50] = 100  # Add spike
    y_clean = remove_spikes(y, zscore_thresh=5.0)
    assert len(y_clean) == len(y)
    assert y_clean[50] < y[50]


def test_hyperspectral_dataset_from_cube():
    """Test HyperspectralDataset creation from cube."""
    wn = np.linspace(500, 1500, 50)
    cube = np.random.randn(10, 12, 50)
    meta = pd.DataFrame({"id": [1]})

    hsd = HyperspectralDataset.from_cube(cube, wn, meta, {"name": "test"})
    assert hsd.wavenumbers.shape == wn.shape
    assert hsd.spectra.shape[1] == len(wn)


def test_hyperspectral_dataset_to_cube():
    """Test converting back to cube."""
    wn = np.linspace(500, 1500, 50)
    cube = np.random.randn(10, 12, 50)
    meta = pd.DataFrame({"id": [1]})

    hsd = HyperspectralDataset.from_cube(cube, wn, meta, {"name": "test"})
    cube_reconstructed = hsd.to_cube()

    assert cube_reconstructed.shape == cube.shape


def test_hyperspectral_dataset_segment():
    """Test hyperspectral segmentation."""
    wn = np.linspace(500, 1500, 50)
    cube = np.random.randn(8, 10, 50)
    meta = pd.DataFrame({"id": [1]})

    hsd = HyperspectralDataset.from_cube(cube, wn, meta, {"name": "test"})

    # Test kmeans segmentation
    labels = hsd.segment(method="kmeans", n_clusters=3)
    assert labels.shape == (8, 10)

    # Test hierarchical segmentation
    labels_hier = hsd.segment(method="hierarchical", n_clusters=3)
    assert labels_hier.shape == (8, 10)

    # Test NMF segmentation
    labels_nmf = hsd.segment(method="nmf", n_clusters=3)
    assert labels_nmf.shape == (8, 10)


def test_preprocessing_config_defaults():
    """Test PreprocessingConfig default values."""
    config = PreprocessingConfig()
    assert config.baseline_method == "als"
    assert config.baseline_enabled is True
    assert config.smoothing_method == "savgol"
    assert config.spike_removal is True


def test_spectral_dataset_to_peaks_edge_cases():
    """Test peak extraction with edge cases."""
    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(3, 100)
    meta = pd.DataFrame({"id": range(3)})
    ds = SpectralDataset(wn, spectra, meta)

    # Peak outside wavenumber range
    peaks = [PeakDefinition(name="out_of_range", column="I_2000", wavenumber=2000, window=(1950, 2050))]
    result = ds.to_peaks(peaks)
    assert "out_of_range" in result.columns
    assert result["out_of_range"].isna().all()

    # Peak with no wavenumber
    peaks_no_wn = [PeakDefinition(name="no_wn", column="I_none", wavenumber=None)]
    result_no_wn = ds.to_peaks(peaks_no_wn)
    assert "no_wn" not in result_no_wn.columns
