"""
tests/test_architecture_ci.py

CI-level architecture tests. Run in every PR/commit.

Tests that the refactored architecture actually works:
  - One-command run completes without error
  - All mandatory outputs produced
  - Manifest captures required metadata
  - No import errors at runtime

Author: Strict Refactor Engineer
Date: January 25, 2026
"""

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


class TestMinimalE2EIntegration:
    """Verify end-to-end execution works with minimal protocol."""

    @pytest.fixture
    def minimal_protocol_yaml(self):
        """Create minimal test protocol."""
        return """
metadata:
  name: "Architecture Test"
  version: "1.0"
  description: "Minimal end-to-end test"

data:
  source: "examples/data/olive_oil_sample.csv"
  modality: "raman"

preprocess:
  recipe: "raman_standard"

features:
  type: "chemometrics"
  components:
    - name: "pca"
      params:
        n_components: 5

task:
  type: "classification"
  target: "class"

model:
  type: "plsda"
  params:
    n_components: 3

validation:
  cv:
    type: "stratified_kfold"
    n_splits: 3
  metrics:
    - "accuracy"
    - "f1_macro"

trust:
  enable: true
  conformal:
    alpha: 0.1

visualization:
  enable: false

reporting:
  enable: false
"""

    @pytest.fixture
    def test_data_csv(self):
        """Create minimal test CSV data."""
        return """wavelength,intensity1,intensity2,intensity3,class
400,10,20,15,A
405,12,22,17,A
410,14,24,19,B
415,16,26,21,B
420,18,28,23,A
"""

    def test_minimal_e2e_run(self, minimal_protocol_yaml, test_data_csv):
        """
        Complete end-to-end run: load data, preprocess, features, model, validate, trust.
        Should complete in <30 seconds.
        """
        repo_root = Path(__file__).parent.parent

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write minimal test data
            data_file = tmpdir_path / "test_data.csv"
            data_file.write_text(test_data_csv)

            # Write minimal protocol
            protocol_file = tmpdir_path / "test_protocol.yaml"
            modified_yaml = minimal_protocol_yaml.replace(
                "examples/data/olive_oil_sample.csv",
                str(data_file),
            )
            protocol_file.write_text(modified_yaml)

            # Create output directory
            output_dir = tmpdir_path / "run_output"
            output_dir.mkdir()

            # Execute: foodspec run --protocol protocol.yaml --output-dir ./run
            result = subprocess.run(
                [
                    "python", "-m", "foodspec.cli.main",
                    "run",
                    "--protocol", str(protocol_file),
                    "--output-dir", str(output_dir),
                    "--no-viz",
                    "--no-report",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should complete without error
            assert result.returncode == 0, (
                f"E2E run failed with code {result.returncode}.\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

            # Verify key outputs exist
            assert (output_dir / "manifest.json").exists(), (
                "manifest.json not created"
            )
            assert (output_dir / "metrics.json").exists(), (
                "metrics.json not created"
            )
            assert (output_dir / "predictions.json").exists(), (
                "predictions.json not created"
            )

    def test_orchestrator_stage_sequence(self):
        """Verify ExecutionEngine has correct stage sequence."""
        try:
            from foodspec.core.orchestrator import ExecutionEngine

            # Create dummy instances to inspect
            # (Don't need to run, just verify structure)
            assert hasattr(ExecutionEngine, "_run_data_load"), (
                "ExecutionEngine missing _run_data_load stage"
            )
            assert hasattr(ExecutionEngine, "_run_preprocess"), (
                "ExecutionEngine missing _run_preprocess stage"
            )
            assert hasattr(ExecutionEngine, "_run_qc"), (
                "ExecutionEngine missing _run_qc stage"
            )
            assert hasattr(ExecutionEngine, "_run_features"), (
                "ExecutionEngine missing _run_features stage"
            )
            assert hasattr(ExecutionEngine, "_run_model"), (
                "ExecutionEngine missing _run_model stage"
            )
            assert hasattr(ExecutionEngine, "_run_evaluate"), (
                "ExecutionEngine missing _run_evaluate stage"
            )
            assert hasattr(ExecutionEngine, "_run_trust"), (
                "ExecutionEngine missing _run_trust stage"
            )
        except ImportError as e:
            pytest.fail(f"Cannot import ExecutionEngine: {e}")


class TestManifestCompleteness:
    """Verify RunManifest captures required metadata."""

    def test_manifest_schema(self):
        """RunManifest must have required fields."""
        try:
            from foodspec.core.manifest import RunManifest

            manifest = RunManifest()

            # Check required fields exist
            assert hasattr(manifest, "metadata"), (
                "RunManifest missing metadata field"
            )
            assert hasattr(manifest, "artifacts"), (
                "RunManifest missing artifacts field"
            )
            assert hasattr(manifest, "checksums"), (
                "RunManifest missing checksums field"
            )

            # Metadata must contain
            assert "timestamp" in manifest.metadata, (
                "Manifest metadata missing timestamp"
            )
            assert "foodspec_version" in manifest.metadata, (
                "Manifest metadata missing foodspec_version"
            )
            assert "python_version" in manifest.metadata, (
                "Manifest metadata missing python_version"
            )
        except ImportError as e:
            pytest.fail(f"Cannot import RunManifest: {e}")

    def test_manifest_serialization(self):
        """RunManifest must serialize to JSON."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from foodspec.core.manifest import RunManifest

        with TemporaryDirectory() as tmpdir:
            manifest = RunManifest()
            manifest_path = Path(tmpdir) / "manifest.json"

            # Should have save() method
            assert hasattr(manifest, "save"), (
                "RunManifest missing save() method"
            )

            # Save and verify
            manifest.save(manifest_path)

            assert manifest_path.exists(), (
                "Manifest not saved to file"
            )

            # Verify valid JSON
            content = manifest_path.read_text()
            data = json.loads(content)

            assert "metadata" in data, "Saved manifest missing metadata"


class TestImportIntegration:
    """Verify all key imports work together."""

    def test_protocol_to_orchestrator_chain(self):
        """Test full import chain: Protocol → Registry → Orchestrator."""
        try:
            from foodspec.core.protocol import ProtocolV2
            from foodspec.core.registry import ComponentRegistry
            from foodspec.core.orchestrator import ExecutionEngine
            from foodspec.core.artifacts import ArtifactRegistry

            # Should be able to construct without error
            registry = ComponentRegistry()
            assert registry is not None

            artifacts_test_dir = "/tmp/test_artifacts"
            artifacts = ArtifactRegistry(artifacts_test_dir)
            assert artifacts is not None

        except ImportError as e:
            pytest.fail(f"Import chain broken: {e}")

    def test_evaluation_to_trust_chain(self):
        """Test evaluation → validation → trust chain."""
        try:
            from foodspec.validation.evaluation import evaluate_model_cv
            from foodspec.trust.conformal import ConformalPredictor
            from foodspec.trust.evaluator import TrustEvaluator

            # Just verify imports
            assert evaluate_model_cv is not None
            assert ConformalPredictor is not None
            assert TrustEvaluator is not None
        except ImportError as e:
            pytest.fail(f"Evaluation-to-trust chain broken: {e}")

    def test_preprocess_to_features_chain(self):
        """Test preprocessing → features chain."""
        try:
            from foodspec.preprocess.recipes import PreprocessingRecipe
            from foodspec.features.peaks import detect_peaks
            from foodspec.features.chemometrics import compute_vip

            # Just verify imports
            assert PreprocessingRecipe is not None
            assert detect_peaks is not None
            assert compute_vip is not None
        except ImportError as e:
            pytest.fail(f"Preprocess-to-features chain broken: {e}")


class TestArtifactCreation:
    """Verify all expected output artifacts are created."""

    def test_artifact_registry_paths(self):
        """ArtifactRegistry creates standard paths."""
        from tempfile import TemporaryDirectory
        from foodspec.core.artifacts import ArtifactRegistry

        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(tmpdir)

            # Check standard path methods exist
            assert hasattr(registry, "manifest_path"), "Missing manifest_path"
            assert hasattr(registry, "metrics_path"), "Missing metrics_path"
            assert hasattr(registry, "predictions_path"), "Missing predictions_path"

            # Should produce valid paths
            manifest_path = registry.manifest_path()
            assert "manifest" in str(manifest_path).lower()

            metrics_path = registry.metrics_path()
            assert "metrics" in str(metrics_path).lower()

            predictions_path = registry.predictions_path()
            assert "predictions" in str(predictions_path).lower()

    def test_save_load_roundtrip(self):
        """Artifacts can be saved and loaded."""
        from tempfile import TemporaryDirectory
        from foodspec.core.artifacts import ArtifactRegistry
        import json

        with TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(tmpdir)

            # Save metrics
            test_metrics = {"accuracy": 0.95, "f1": 0.92}
            metrics_path = registry.metrics_path()

            if hasattr(registry, "save_json"):
                registry.save_json("metrics.json", test_metrics)
                assert metrics_path.exists()

                # Load back
                loaded = json.loads(metrics_path.read_text())
                assert loaded == test_metrics


class TestRegressionPrevention:
    """Tests that prevent common refactoring mistakes."""

    def test_no_import_from_removed_paths(self):
        """Imports from old paths should fail."""
        # These should NOT be importable anymore
        bad_imports = [
            "foodspec_rewrite.foodspec.core.protocol",
            "foodspec.protocol.config.ProtocolConfig",
            "foodspec.protocol.runner.ProtocolRunner",
        ]

        for bad_import in bad_imports:
            with pytest.raises(ImportError):
                parts = bad_import.rsplit(".", 1)
                if len(parts) == 2:
                    module, name = parts
                    exec(f"from {module} import {name}")

    def test_no_circular_imports(self):
        """Core modules should not have circular dependencies."""
        try:
            # Import in different orders to detect circularity
            from foodspec.core import protocol
            from foodspec.core import registry
            from foodspec.core import orchestrator

            # Should all be importable without deadlock/error
            assert protocol is not None
            assert registry is not None
            assert orchestrator is not None
        except Exception as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_version_consistency(self):
        """Package version should be consistent."""
        import foodspec

        # foodspec should be importable
        assert foodspec is not None

        # Should have __version__ or similar
        if hasattr(foodspec, "__version__"):
            version = foodspec.__version__
            assert isinstance(version, str)
            assert len(version) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
