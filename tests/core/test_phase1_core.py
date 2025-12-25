"""Tests for Phase 1: Core objects and unified entry point."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from foodspec import FoodSpec, OutputBundle, RunRecord, Spectrum, FoodSpectrumSet


class TestSpectrum:
    """Test Spectrum core object."""
    
    def test_spectrum_creation(self):
        """Test creating a Spectrum."""
        x = np.linspace(500, 1000, 100)
        y = np.random.randn(100)
        
        spec = Spectrum(x=x, y=y, kind="raman", metadata={"sample_id": "s1"})
        
        assert spec.n_points == 100
        assert spec.kind == "raman"
        assert spec.x_unit == "cm-1"
        assert spec.metadata["sample_id"] == "s1"
    
    def test_spectrum_normalization(self):
        """Test spectrum normalization methods."""
        x = np.linspace(500, 1000, 100)
        y = np.ones(100) * 10.0  # All ones scaled
        
        spec = Spectrum(x=x, y=y, kind="raman")
        
        # Vector normalization
        spec_norm = spec.normalize("vector")
        assert np.isclose(np.linalg.norm(spec_norm.y), 1.0)
        
        # Max normalization
        spec_max = spec.normalize("max")
        assert np.isclose(np.max(np.abs(spec_max.y)), 1.0)
    
    def test_spectrum_crop(self):
        """Test spectrum cropping."""
        x = np.linspace(500, 1000, 100)
        y = np.random.randn(100)
        spec = Spectrum(x=x, y=y, kind="raman")
        
        cropped = spec.crop_wavenumber(600, 800)
        assert cropped.n_points < spec.n_points
        assert cropped.x.min() >= 600
        assert cropped.x.max() <= 800
    
    def test_spectrum_config_hash(self):
        """Test config hash for reproducibility."""
        spec1 = Spectrum(x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), kind="raman", metadata={"id": "1"})
        spec2 = Spectrum(x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), kind="raman", metadata={"id": "1"})
        spec3 = Spectrum(x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), kind="raman", metadata={"id": "2"})
        
        assert spec1.config_hash == spec2.config_hash
        assert spec1.config_hash != spec3.config_hash


class TestRunRecord:
    """Test RunRecord for provenance tracking."""
    
    def test_run_record_creation(self):
        """Test creating a RunRecord."""
        config = {"algorithm": "rf", "n_estimators": 100}
        record = RunRecord(
            workflow_name="oil_auth",
            config=config,
            dataset_hash="abc12345",
        )
        
        assert record.workflow_name == "oil_auth"
        assert record.config == config
        assert len(record.config_hash) == 8
        assert len(record.run_id) > 0
    
    def test_run_record_add_step(self):
        """Test adding steps to a run record."""
        record = RunRecord(
            workflow_name="test",
            config={},
            dataset_hash="xyz",
        )
        
        record.add_step("preprocessing", "step_hash_1")
        record.add_step("feature_extraction", "step_hash_2", metadata={"n_features": 50})
        
        assert len(record.step_records) == 2
        assert record.step_records[0]["name"] == "preprocessing"
        assert record.step_records[1]["metadata"]["n_features"] == 50
    
    def test_run_record_serialization(self):
        """Test run record JSON serialization."""
        with TemporaryDirectory() as tmpdir:
            record = RunRecord(
                workflow_name="test",
                config={"key": "value"},
                dataset_hash="hash123",
            )
            
            path = Path(tmpdir) / "record.json"
            record.to_json(path)
            
            assert path.exists()
            
            # Load and verify
            loaded = RunRecord.from_json(path)
            assert loaded.workflow_name == "test"
            assert loaded.config == {"key": "value"}


class TestOutputBundle:
    """Test OutputBundle for artifact management."""
    
    def test_output_bundle_creation(self):
        """Test creating an OutputBundle."""
        record = RunRecord("test", {}, "hash")
        bundle = OutputBundle(run_record=record)
        
        assert bundle.run_record == record
        assert len(bundle.metrics) == 0
        assert len(bundle.diagnostics) == 0
        assert len(bundle.artifacts) == 0
    
    def test_output_bundle_add_items(self):
        """Test adding metrics, diagnostics, artifacts."""
        record = RunRecord("test", {}, "hash")
        bundle = OutputBundle(run_record=record)
        
        # Add metrics
        bundle.add_metrics("accuracy", 0.95)
        bundle.add_metrics("f1", 0.93)
        
        # Add diagnostics
        bundle.add_diagnostic("confusion_matrix", np.array([[10, 2], [1, 12]]))
        
        # Add artifacts
        model = {"name": "dummy_model"}
        bundle.add_artifact("model", model)
        
        assert bundle.metrics["accuracy"] == 0.95
        assert bundle.diagnostics["confusion_matrix"].shape == (2, 2)
        assert bundle.artifacts["model"] == model
    
    def test_output_bundle_export(self):
        """Test exporting bundle to disk."""
        with TemporaryDirectory() as tmpdir:
            record = RunRecord("test", {"param": "value"}, "hash123")
            bundle = OutputBundle(run_record=record)
            
            # Add some outputs
            bundle.add_metrics("accuracy", 0.92)
            bundle.add_metrics("cv_scores", pd.DataFrame({"fold": [1, 2, 3], "score": [0.9, 0.92, 0.94]}))
            
            # Export
            out_dir = bundle.export(tmpdir)
            
            assert (out_dir / "metrics" / "metrics.json").exists()
            assert (out_dir / "metrics" / "cv_scores.csv").exists()
            assert (out_dir / "provenance.json").exists()


class TestFoodSpec:
    """Test FoodSpec unified entry point."""
    
    def test_foodspec_from_array(self):
        """Test FoodSpec initialization from numpy array."""
        x = np.random.randn(10, 50)  # 10 samples, 50 wavenumbers
        wn = np.linspace(500, 1000, 50)
        meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(10)]})
        
        fs = FoodSpec(x, wavenumbers=wn, metadata=meta, modality="raman")
        
        assert len(fs.data) == 10
        assert fs.data.x.shape[1] == 50
        assert fs.bundle is not None
    
    def test_foodspec_from_foodspectrumset(self):
        """Test FoodSpec initialization from FoodSpectrumSet."""
        x = np.random.randn(5, 30)
        wn = np.linspace(500, 1000, 30)
        meta = pd.DataFrame({"label": ["A", "B", "A", "B", "A"]})
        
        fss = FoodSpectrumSet(x=x, wavenumbers=wn, metadata=meta, modality="ftir")
        fs = FoodSpec(fss)
        
        assert len(fs.data) == 5
        assert fs.data.modality == "ftir"
    
    def test_foodspec_chainable_api(self):
        """Test that FoodSpec methods are chainable."""
        x = np.random.randn(10, 50)
        wn = np.linspace(500, 1000, 50)
        meta = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(10)],
            "label": ["A", "B"] * 5,
        })
        
        fs = FoodSpec(x, wavenumbers=wn, metadata=meta)
        
        # Chain methods
        result = fs.qc().preprocess("standard").features("standard")
        
        assert isinstance(result, FoodSpec)
        assert len(fs._steps_applied) > 0
    
    def test_foodspec_summary(self):
        """Test FoodSpec summary generation."""
        x = np.random.randn(5, 20)
        wn = np.linspace(500, 1000, 20)
        
        fs = FoodSpec(x, wavenumbers=wn)
        fs.qc().preprocess("quick")
        
        summary = fs.summary()
        
        assert "FoodSpec" in summary
        assert "n=5" in summary
        assert "qc" in summary
    
    def test_foodspec_export(self):
        """Test FoodSpec export functionality."""
        with TemporaryDirectory() as tmpdir:
            x = np.random.randn(8, 30)
            wn = np.linspace(500, 1000, 30)
            meta = pd.DataFrame({"label": ["A", "B", "A", "B", "A", "B", "A", "B"]})
            
            fs = FoodSpec(x, wavenumbers=wn, metadata=meta)
            fs.bundle.add_metrics("test_metric", 0.95)
            fs.bundle.add_metrics("test_df", pd.DataFrame({"a": [1, 2, 3]}))
            
            out_dir = fs.export(tmpdir)
            
            assert (out_dir / "metrics").exists()
            assert (out_dir / "provenance.json").exists()


class TestPhase1Integration:
    """Integration tests for Phase 1 components."""
    
    def test_end_to_end_workflow(self):
        """Test complete Phase 1 workflow: load -> QC -> preprocess -> export."""
        with TemporaryDirectory() as tmpdir:
            # Create synthetic data
            n_samples = 15
            n_wn = 100
            x = np.random.randn(n_samples, n_wn)
            wn = np.linspace(500, 2000, n_wn)
            meta = pd.DataFrame({
                "sample_id": [f"sample_{i:03d}" for i in range(n_samples)],
                "oil_type": ["olive", "sunflower", "canola"] * 5,
                "batch": [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3],
            })
            
            # Initialize FoodSpec
            fs = FoodSpec(
                x,
                wavenumbers=wn,
                metadata=meta,
                modality="raman",
                kind="oils",
                output_dir=tmpdir,
            )
            
            # Chain operations
            fs.qc(method="isolation_forest", threshold=0.5)
            fs.preprocess(preset="standard")
            
            # Check internal state
            assert "qc" in fs._steps_applied
            assert "preprocess(standard)" in fs._steps_applied
            
            # Export
            out_dir = fs.export()
            
            # Verify outputs
            assert out_dir.exists()
            assert (out_dir / "provenance.json").exists()
            
            # Verify provenance
            with open(out_dir / "provenance.json") as f:
                prov = json.load(f)
            
            assert prov["workflow_name"] == "foodspec"
            assert len(prov["step_records"]) == 2  # qc + preprocess


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
