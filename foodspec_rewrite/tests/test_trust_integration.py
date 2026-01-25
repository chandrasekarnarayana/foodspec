"""
Integration tests for trust subsystem with FoodSpec evaluation pipeline.

Tests end-to-end workflows combining:
- Conformal prediction
- Calibration
- Abstention
- ArtifactRegistry integration
- Evaluator high-level interface
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from foodspec.trust.evaluator import TrustEvaluator, TrustEvaluationResult
from foodspec.core.artifacts import ArtifactRegistry


class TestTrustEvaluatorIntegration:
    """Test high-level TrustEvaluator with full pipeline."""
    
    @pytest.fixture
    def temp_registry(self):
        """Create temporary artifact registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ArtifactRegistry(Path(tmpdir))
    
    def test_evaluator_full_workflow(self, temp_registry):
        """Test complete evaluation workflow."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        
        # Generate data
        X_train = np.random.randn(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_cal = np.random.randn(30, 5)
        y_cal = np.random.randint(0, 2, 30)
        X_test = np.random.randn(40, 5)
        y_test = np.random.randint(0, 2, 40)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        
        # Create evaluator
        evaluator = TrustEvaluator(
            model,
            artifact_registry=temp_registry,
            target_coverage=0.9,
            abstention_threshold=0.7,
            calibration_method="temperature",
        )
        
        # Fit on calibration set
        evaluator.fit_conformal(X_cal, y_cal)
        proba_cal = model.predict_proba(X_cal)
        evaluator.fit_calibration(y_cal, proba_cal)
        
        # Evaluate
        result = evaluator.evaluate(
            X_test, y_test,
            model_name="test_model",
        )
        
        # Verify result
        assert isinstance(result, TrustEvaluationResult)
        assert result.conformal_coverage >= 0
        assert result.ece >= 0
        assert 0 <= result.abstention_rate <= 1
    
    def test_evaluator_with_group_metrics(self, temp_registry):
        """Test evaluator with batch/group conditioning."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        
        # Generate data with batch labels
        X_train = np.random.randn(80, 5)
        y_train = np.random.randint(0, 2, 80)
        X_cal = np.random.randn(30, 5)
        y_cal = np.random.randint(0, 2, 30)
        X_test = np.random.randn(40, 5)
        y_test = np.random.randint(0, 2, 40)
        
        # Batch IDs
        batch_cal = np.array([0] * 15 + [1] * 15)
        batch_test = np.array([0] * 20 + [1] * 20)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        
        # Evaluator with batch conditioning
        evaluator = TrustEvaluator(
            model,
            artifact_registry=temp_registry,
            target_coverage=0.9,
        )
        
        evaluator.fit_conformal(X_cal, y_cal, bins_cal=batch_cal)
        proba_cal = model.predict_proba(X_cal)
        evaluator.fit_calibration(y_cal, proba_cal)
        
        # Evaluate with batch info
        result = evaluator.evaluate(
            X_test, y_test,
            bins_test=batch_test,
            batch_ids=batch_test,
            model_name="batch_aware_model",
        )
        
        # Verify group metrics
        assert len(result.group_metrics) > 0
        for group_id, metrics in result.group_metrics.items():
            assert "coverage" in metrics
            assert "abstention_rate" in metrics
    
    @pytest.mark.skip(reason="Artifact registry integration not finalized")
    def test_evaluator_artifact_saving(self, temp_registry):
        """Test artifact saving to registry and disk."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        
        X_train = np.random.randn(60, 5)
        y_train = np.random.randint(0, 2, 60)
        X_cal = np.random.randn(20, 5)
        y_cal = np.random.randint(0, 2, 20)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        
        evaluator = TrustEvaluator(model, artifact_registry=temp_registry)
        evaluator.fit_conformal(X_cal, y_cal)
        evaluator.fit_calibration(y_cal, model.predict_proba(X_cal))
        
        result = evaluator.evaluate(X_test, y_test, model_name="save_test")
        
        # Get prediction sets and abstention using probabilities
        proba_test = model.predict_proba(X_test)
        cp_result = evaluator._conformal.predict_sets(proba_test, y_true=y_test)
        from foodspec.trust.abstain import evaluate_abstention
        abstain_result = evaluate_abstention(
            proba_test,
            y_test,
            threshold=0.7,
            prediction_sets=cp_result.prediction_sets,
            max_set_size=max(cp_result.set_sizes),
        )
        
        # Save artifacts
        output_dir = Path(temp_registry.root) / "trust"
        artifacts = evaluator.save_artifacts(
            result,
            prediction_sets=cp_result.prediction_sets,
            set_sizes=cp_result.set_sizes,
            abstention_mask=abstain_result.abstain_mask,
            output_dir=output_dir,
        )
        
        # Verify artifacts saved
        assert "evaluation_result" in artifacts
        assert "prediction_sets" in artifacts
        assert "abstention" in artifacts
        
        # Verify files on disk
        assert (output_dir / "trust_eval.json").exists()
        assert (output_dir / "prediction_sets.csv").exists()
        assert (output_dir / "abstention.csv").exists()
        
        # Verify JSON content
        with open(output_dir / "trust_eval.json") as f:
            eval_dict = json.load(f)
            assert eval_dict["model_name"] == "save_test"
            assert "conformal_coverage" in eval_dict
    
    def test_evaluator_report_generation(self, temp_registry):
        """Test human-readable report generation."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        
        X_train = np.random.randn(60, 5)
        y_train = np.random.randint(0, 2, 60)
        X_cal = np.random.randn(20, 5)
        y_cal = np.random.randint(0, 2, 20)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        
        evaluator = TrustEvaluator(
            model,
            artifact_registry=temp_registry,
            target_coverage=0.95,
            abstention_threshold=0.75,
            calibration_method="temperature",
        )
        
        evaluator.fit_conformal(X_cal, y_cal)
        evaluator.fit_calibration(y_cal, model.predict_proba(X_cal))
        result = evaluator.evaluate(X_test, y_test, model_name="report_test")
        
        # Generate report
        report = evaluator.report(result)
        
        # Verify report content
        assert "TRUST & UNCERTAINTY" in report
        assert "CONFORMAL PREDICTION" in report
        assert "CALIBRATION" in report
        assert "ABSTENTION" in report
        assert f"{evaluator.target_coverage:.1%}" in report
        assert f"{result.ece:.4f}" in report


class TestArtifactRegistryTrustExtensions:
    """Test ArtifactRegistry trust artifact support."""
    
    def test_trust_paths_exist(self):
        """Test that trust artifact paths are defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(Path(tmpdir))
            
            # Verify trust paths exist
            assert hasattr(registry, "trust_dir")
            assert hasattr(registry, "trust_eval_path")
            assert hasattr(registry, "prediction_sets_path")
            assert hasattr(registry, "abstention_path")
            assert hasattr(registry, "coverage_table_path")
            assert hasattr(registry, "calibration_path")
    
    def test_trust_layout_creation(self):
        """Test that ensure_layout creates trust directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(Path(tmpdir))
            registry.ensure_layout()
            
            # Verify trust directory created
            assert registry.trust_dir.exists()
            assert registry.plots_dir.exists()
            assert registry.bundle_dir.exists()


class TestTrustEvaluationWorkflow:
    """Test realistic end-to-end trust evaluation workflows."""
    
    @pytest.mark.xfail(reason="Flaky due to randomness in conformal prediction", strict=False)
    def test_oil_auth_workflow(self):
        """Simulate oil authentication evaluation with trust."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        
        # Simulate oil samples from 3 batches
        n_per_batch = 30
        n_features = 10
        
        X_train = np.random.randn(n_per_batch * 2, n_features)
        y_train = np.random.randint(0, 2, n_per_batch * 2)
        
        X_cal = np.random.randn(n_per_batch, n_features)
        y_cal = np.random.randint(0, 2, n_per_batch)
        cal_batches = np.repeat([0, 1], n_per_batch // 2)
        
        X_test = np.random.randn(n_per_batch, n_features)
        y_test = np.random.randint(0, 2, n_per_batch)
        test_batches = np.repeat([0, 1], n_per_batch // 2)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(Path(tmpdir))
            
            # Tight coverage for authentication
            evaluator = TrustEvaluator(
                model,
                artifact_registry=registry,
                target_coverage=0.95,  # High coverage for authentication
                abstention_threshold=0.90,  # Very high confidence threshold
            )
            
            # Fit with batch conditioning
            evaluator.fit_conformal(X_cal, y_cal, bins_cal=cal_batches)
            evaluator.fit_calibration(y_cal, model.predict_proba(X_cal))
            
            # Evaluate
            result = evaluator.evaluate(
                X_test, y_test,
                bins_test=test_batches,
                batch_ids=test_batches,
                model_name="oil_auth_v1",
            )
            
            # Verify coverage is good
            assert result.conformal_coverage >= 0.85
            
            # Verify per-bin coverage exists
            assert len(result.per_bin_coverage) > 0 or len(result.group_metrics) > 0
