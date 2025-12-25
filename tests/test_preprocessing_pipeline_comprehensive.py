"""Comprehensive tests for preprocessing_pipeline to increase coverage."""
import numpy as np
import pandas as pd
import pytest

from foodspec.preprocessing_pipeline import (
    detect_input_mode,
    baseline_als,
    extract_peaks_from_spectra,
    run_full_preprocessing,
    PreprocessingConfig,
)
from foodspec.rq import PeakDefinition


def test_detect_input_mode_peak_table():
    """Test detection of peak table format."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "I_1000": [10.0, 12.0, 11.0],
        "I_1200": [8.0, 9.0, 8.5],
    })
    
    mode = detect_input_mode(df)
    assert mode == "peak_table"


def test_detect_input_mode_raw_spectra():
    """Test detection of raw spectra format."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "500.0": [1.0, 1.1, 1.2],
        "600.0": [2.0, 2.1, 2.2],
        "700.0": [3.0, 3.1, 3.2],
        "800.0": [4.0, 4.1, 4.2],
    })
    
    mode = detect_input_mode(df)
    assert mode == "raw_spectra"


def test_detect_input_mode_fallback():
    """Test fallback to peak_table."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "value": [10.0, 12.0, 11.0],
    })
    
    mode = detect_input_mode(df)
    assert mode == "peak_table"


def test_baseline_als_correction():
    """Test ALS baseline correction."""
    y = np.linspace(0, 10, 100) + np.random.randn(100) * 0.1
    baseline = baseline_als(y, lam=1e5, p=0.01, niter=10)
    
    assert len(baseline) == len(y)
    # Baseline may have small negative values due to interpolation
    assert baseline.min() > -1.0


def test_extract_peaks_from_spectra():
    """Test peak extraction from raw spectra."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "500.0": [1.0, 1.1, 1.2],
        "600.0": [2.0, 2.1, 2.2],
        "700.0": [3.0, 3.1, 3.2],
        "800.0": [4.0, 4.1, 4.2],
        "900.0": [5.0, 5.1, 5.2],
        "1000.0": [6.0, 6.1, 6.2],
    })
    
    peaks = [
        PeakDefinition(name="peak1", column="I_700", wavenumber=700, window=(650, 750)),
        PeakDefinition(name="peak2", column="I_900", wavenumber=900, window=(850, 950)),
    ]
    
    wavenumber_cols = ["500.0", "600.0", "700.0", "800.0", "900.0", "1000.0"]
    result = extract_peaks_from_spectra(df, peaks, wavenumber_cols)
    
    assert "peak1" in result.columns
    assert "peak2" in result.columns
    assert "sample_id" in result.columns
    assert len(result) == 3


def test_extract_peaks_peak_outside_range():
    """Test peak extraction when peak is outside wavenumber range."""
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "500.0": [1.0, 1.1],
        "600.0": [2.0, 2.1],
        "700.0": [3.0, 3.1],
    })
    
    peaks = [PeakDefinition(name="out_of_range", column="I_2000", wavenumber=2000, window=(1950, 2050))]
    wavenumber_cols = ["500.0", "600.0", "700.0"]
    
    result = extract_peaks_from_spectra(df, peaks, wavenumber_cols)
    
    assert "out_of_range" in result.columns
    assert result["out_of_range"].isna().all()


def test_extract_peaks_no_wavenumber():
    """Test peak extraction when peak has no wavenumber."""
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "500.0": [1.0, 1.1],
        "600.0": [2.0, 2.1],
    })
    
    peaks = [PeakDefinition(name="no_wn", column="I_none", wavenumber=None)]
    wavenumber_cols = ["500.0", "600.0"]
    
    result = extract_peaks_from_spectra(df, peaks, wavenumber_cols)
    
    # Peak with no wavenumber should be skipped
    assert "no_wn" not in result.columns


def test_run_full_preprocessing_peak_table():
    """Test full preprocessing with peak table input."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "I_1000": [10.0, 12.0, 11.0],
        "I_1200": [8.0, 9.0, 8.5],
    })
    
    config = PreprocessingConfig()
    result = run_full_preprocessing(df, config)
    
    assert "sample_id" in result.columns
    assert "I_1000" in result.columns
    assert len(result) == 3


def test_run_full_preprocessing_raw_spectra():
    """Test full preprocessing with raw spectra input."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "500.0": [1.0, 1.1, 1.2],
        "600.0": [2.0, 2.1, 2.2],
        "700.0": [3.0, 3.1, 3.2],
        "800.0": [4.0, 4.1, 4.2],
        "900.0": [5.0, 5.1, 5.2],
    })
    
    peaks = [
        PeakDefinition(name="peak1", column="I_700", wavenumber=700, window=(650, 750)),
    ]
    
    config = PreprocessingConfig(
        baseline_method="als",
        smoothing_method="savgol",
        normalization="vector",
        peak_definitions=peaks,
    )
    
    result = run_full_preprocessing(df, config)
    
    assert "sample_id" in result.columns
    assert "peak1" in result.columns
    assert len(result) == 3


def test_run_full_preprocessing_with_spike_removal():
    """Test preprocessing with spike removal enabled."""
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "500.0": [1.0, 1.1],
        "600.0": [100.0, 2.1],  # Spike in first sample
        "700.0": [3.0, 3.1],
        "800.0": [4.0, 4.1],
    })
    
    peaks = [PeakDefinition(name="peak1", column="I_700", wavenumber=700, window=(650, 750))]
    
    config = PreprocessingConfig(
        spike_removal=True,
        spike_zscore_thresh=3.0,
        peak_definitions=peaks,
    )
    
    result = run_full_preprocessing(df, config)
    assert len(result) == 2


def test_run_full_preprocessing_different_baseline_methods():
    """Test preprocessing with different baseline methods."""
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "500.0": [1.0, 1.1],
        "600.0": [2.0, 2.1],
        "700.0": [3.0, 3.1],
        "800.0": [4.0, 4.1],
    })
    
    peaks = [PeakDefinition(name="peak1", column="I_700", wavenumber=700, window=(650, 750))]
    
    # Test with rubberband baseline
    config_rb = PreprocessingConfig(
        baseline_method="rubberband",
        peak_definitions=peaks,
    )
    result_rb = run_full_preprocessing(df, config_rb)
    assert len(result_rb) == 2
    
    # Test with polynomial baseline
    config_poly = PreprocessingConfig(
        baseline_method="polynomial",
        baseline_order=3,
        peak_definitions=peaks,
    )
    result_poly = run_full_preprocessing(df, config_poly)
    assert len(result_poly) == 2
    
    # Test with no baseline
    config_none = PreprocessingConfig(
        baseline_method="none",
        peak_definitions=peaks,
    )
    result_none = run_full_preprocessing(df, config_none)
    assert len(result_none) == 2


def test_run_full_preprocessing_different_normalizations():
    """Test preprocessing with different normalization modes."""
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "500.0": [1.0, 1.1],
        "600.0": [2.0, 2.1],
        "700.0": [3.0, 3.1],
    })
    
    peaks = [PeakDefinition(name="peak1", column="I_600", wavenumber=600, window=(550, 650))]
    
    for norm_mode in ["vector", "area", "none"]:
        config = PreprocessingConfig(
            normalization=norm_mode,
            peak_definitions=peaks,
        )
        result = run_full_preprocessing(df, config)
        assert len(result) == 2
        assert "peak1" in result.columns


def test_preprocessing_config_custom_values():
    """Test PreprocessingConfig with custom values."""
    config = PreprocessingConfig(
        baseline_method="polynomial",
        baseline_lambda=1e6,
        baseline_p=0.05,
        baseline_order=4,
        smoothing_window=11,
        smoothing_polyorder=4,
        normalization="area",
        spike_zscore_thresh=10.0,
    )
    
    assert config.baseline_method == "polynomial"
    assert config.baseline_lambda == 1e6
    assert config.baseline_order == 4
    assert config.smoothing_window == 11
    assert config.normalization == "area"
