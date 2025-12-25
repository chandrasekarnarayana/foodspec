"""Comprehensive tests for high-value modules to reach 90% coverage."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from foodspec.core.api import FoodSpec
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.rq import RatioQualityEngine, RatioDefinition, PeakDefinition, RQConfig


def test_foodspec_init_with_dataset():
    """Test FoodSpec initialization with FoodSpectrumSet."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5)})
    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    
    fs = FoodSpec(ds, modality="raman", kind="test")
    assert fs.data.x.shape == (5, 100)
    assert fs.config["modality"] == "raman"


def test_foodspec_init_with_numpy():
    """Test FoodSpec initialization with numpy array."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5)})
    
    fs = FoodSpec(x, wavenumbers=wn, metadata=meta, modality="raman")
    assert fs.data.x.shape == (5, 100)
    assert len(fs.data.metadata) == 5


def test_foodspec_init_with_dataframe():
    """Test FoodSpec initialization with DataFrame."""
    df = pd.DataFrame({
        "wavenumber": [500, 600, 700, 800, 900],
        "sample1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "sample2": [1.5, 2.5, 3.5, 4.5, 5.5],
    })
    
    fs = FoodSpec(df, modality="raman")
    assert fs.data.x.shape[0] >= 1  # At least one sample


def test_rq_engine_initialization():
    """Test RatioQualityEngine initialization."""
    peaks = [PeakDefinition(name="peak1", column="I_1000", wavenumber=1000)]
    ratios = [RatioDefinition(name="ratio1", numerator="I_1000", denominator="I_1200")]
    config = RQConfig(oil_col="Oil_Name", matrix_col="matrix")
    
    engine = RatioQualityEngine(peaks, ratios, config)
    assert engine.config.oil_col == "Oil_Name"
    assert len(engine.peaks) == 1
    assert len(engine.ratios) == 1


def test_rq_engine_add_peak_definitions():
    """Test adding peak definitions to RQ engine."""
    peaks = [
        PeakDefinition(name="peak1", column="I_1000", wavenumber=1000),
        PeakDefinition(name="peak2", column="I_1200", wavenumber=1200),
    ]
    ratios = []
    
    engine = RatioQualityEngine(peaks, ratios)
    assert len(engine.peaks) == 2


def test_rq_engine_add_ratio_definitions():
    """Test adding ratio definitions to RQ engine."""
    peaks = []
    ratios = [
        RatioDefinition(name="ratio1", numerator="peak1", denominator="peak2"),
        RatioDefinition(name="ratio2", numerator="peak3", denominator="peak4"),
    ]
    
    engine = RatioQualityEngine(peaks, ratios)
    assert len(engine.ratios) == 2


def test_rq_engine_compute_ratios():
    """Test computing ratios from peak table."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "I_1000": [10.0, 12.0, 11.0],
        "I_1200": [5.0, 6.0, 5.5],
        "I_1400": [8.0, 9.0, 8.5],
    })
    
    peaks = [
        PeakDefinition(name="peak1", column="I_1000", wavenumber=1000),
        PeakDefinition(name="peak2", column="I_1200", wavenumber=1200),
    ]
    
    ratios = [
        RatioDefinition(name="R_1000_1200", numerator="I_1000", denominator="I_1200"),
    ]
    
    engine = RatioQualityEngine(peaks, ratios)
    
    result = engine.compute_ratios(df)
    
    assert "R_1000_1200" in result.columns
    assert len(result) == 3


def test_ratio_definition():
    """Test RatioDefinition dataclass."""
    ratio = RatioDefinition(
        name="test_ratio",
        numerator="peak1",
        denominator="peak2"
    )
    
    assert ratio.name == "test_ratio"
    assert ratio.numerator == "peak1"
    assert ratio.denominator == "peak2"


def test_peak_definition_defaults():
    """Test PeakDefinition with defaults."""
    peak = PeakDefinition(name="test", column="I_1000")
    
    assert peak.name == "test"
    assert peak.column == "I_1000"
    assert peak.mode == "max"


def test_peak_definition_with_window():
    """Test PeakDefinition with window."""
    peak = PeakDefinition(
        name="test",
        column="I_1000",
        wavenumber=1000,
        window=(950, 1050),
        mode="area"
    )
    
    assert peak.wavenumber == 1000
    assert peak.window == (950, 1050)
    assert peak.mode == "area"


def test_rqconfig_defaults():
    """Test RQConfig default values."""
    config = RQConfig()
    
    assert config.oil_col == "Oil_Name"
    assert config.matrix_col == "matrix"


def test_rqconfig_custom_values():
    """Test RQConfig with custom values."""
    config = RQConfig(
        oil_col="oil_type",
        matrix_col="sample_matrix"
    )
    
    assert config.oil_col == "oil_type"
    assert config.matrix_col == "sample_matrix"


def test_foodspec_output_bundle():
    """Test that FoodSpec creates output bundle."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5)})
    
    fs = FoodSpec(x, wavenumbers=wn, metadata=meta)
    
    assert fs.bundle is not None
    assert fs.bundle.run_record is not None


def test_foodspec_config_tracking():
    """Test that FoodSpec tracks configuration."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5)})
    
    fs = FoodSpec(x, wavenumbers=wn, metadata=meta, modality="ftir", kind="test_data")
    
    assert fs.config["modality"] == "ftir"
    assert fs.config["kind"] == "test_data"


def test_rq_engine_with_empty_dataframe():
    """Test RQ engine with empty peaks/ratios."""
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "I_1000": [10.0, 12.0],
    })
    
    engine = RatioQualityEngine(peaks=[], ratios=[])
    
    # Should return original dataframe
    result = engine.compute_ratios(df)
    assert len(result) == 2


def test_rq_engine_ratio_with_zero_denominator():
    """Test RQ engine handles division by zero."""
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "I_1000": [10.0, 12.0],
        "I_1200": [5.0, 0.0],  # Zero denominator
    })
    
    ratios = [
        RatioDefinition(name="R_test", numerator="I_1000", denominator="I_1200"),
    ]
    
    engine = RatioQualityEngine(peaks=[], ratios=ratios)
    
    result = engine.compute_ratios(df)
    
    # Should handle gracefully (inf or nan)
    assert "R_test" in result.columns


def test_foodspec_data_hash():
    """Test that FoodSpec computes dataset hash."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5)})
    
    fs = FoodSpec(x, wavenumbers=wn, metadata=meta)
    
    assert fs.bundle.run_record.dataset_hash is not None
    assert len(fs.bundle.run_record.dataset_hash) > 0
