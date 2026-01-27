"""Integration tests for ROC/AUC diagnostics in modeling pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from foodspec.modeling.api import fit_predict
from foodspec.modeling.outcome import OutcomeType


class TestROCIntegration:
    """Test ROC integration with fit_predict."""

    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def multiclass_classification_data(self):
        """Generate multiclass classification dataset."""
        X, y = make_classification(
            n_samples=120,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        return X, y

    def test_roc_computed_by_default(self, binary_classification_data):
        """Test that ROC diagnostics are computed by default."""
        X, y = binary_classification_data

        result = fit_predict(
            X,
            y,
            model_name="logistic_regression",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.CLASSIFICATION,
        )

        assert result.roc_diagnostics is not None
        assert result.roc_diagnostics.per_class is not None
        assert len(result.roc_diagnostics.per_class) > 0
        # Check that AUC is available for each class
        for class_key, metrics in result.roc_diagnostics.per_class.items():
            assert metrics.auc is not None
            assert 0 <= metrics.auc <= 1

    def test_roc_can_be_disabled(self, binary_classification_data):
        """Test that ROC computation can be disabled."""
        X, y = binary_classification_data

        result = fit_predict(
            X,
            y,
            model_name="logistic_regression",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.CLASSIFICATION,
            compute_roc=False,
        )

        assert result.roc_diagnostics is None
        assert result.roc_artifacts == {}

    def test_roc_artifacts_saved(self, binary_classification_data):
        """Test that ROC artifacts are saved to disk."""
        X, y = binary_classification_data

        with tempfile.TemporaryDirectory() as tmpdir:
            result = fit_predict(
                X,
                y,
                model_name="logistic_regression",
                scheme="kfold",
                outer_splits=3,
                seed=42,
                allow_random_cv=True,
                outcome_type=OutcomeType.CLASSIFICATION,
                compute_roc=True,
                roc_output_dir=tmpdir,
            )

            # Check artifacts were saved
            assert len(result.roc_artifacts) > 0

            output_path = Path(tmpdir)
            assert (output_path / "tables" / "roc_summary.csv").exists()
            assert (output_path / "json" / "roc_diagnostics.json").exists()

            # Verify artifact paths are recorded with expected keys
            assert "roc_summary" in result.roc_artifacts
            assert "roc_diagnostics_json" in result.roc_artifacts

    def test_roc_multiclass(self, multiclass_classification_data):
        """Test ROC computation for multiclass classification."""
        X, y = multiclass_classification_data

        result = fit_predict(
            X,
            y,
            model_name="logistic_regression",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.CLASSIFICATION,
            compute_roc=True,
        )

        assert result.roc_diagnostics is not None
        # Multiclass should have per-class AUCs
        assert len(result.roc_diagnostics.per_class) >= 1
        # Check micro-averaged metrics
        if result.roc_diagnostics.micro:
            assert result.roc_diagnostics.micro.auc is not None

    def test_roc_respects_seed(self, binary_classification_data):
        """Test that ROC computation is reproducible with same seed."""
        X, y = binary_classification_data

        result1 = fit_predict(
            X,
            y,
            model_name="logistic_regression",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.CLASSIFICATION,
            compute_roc=True,
            roc_n_bootstrap=100,
        )

        result2 = fit_predict(
            X,
            y,
            model_name="logistic_regression",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.CLASSIFICATION,
            compute_roc=True,
            roc_n_bootstrap=100,
        )

        # ROC AUC should be identical (or very close) with same seed
        assert result1.roc_diagnostics is not None
        assert result2.roc_diagnostics is not None
        # Compare AUC values across classes
        for class_key in result1.roc_diagnostics.per_class:
            auc1 = result1.roc_diagnostics.per_class[class_key].auc
            auc2 = result2.roc_diagnostics.per_class[class_key].auc
            assert np.isclose(auc1, auc2, atol=1e-5)

    def test_roc_skipped_for_regression(self):
        """Test that ROC is skipped for regression tasks."""
        X = np.random.randn(100, 20)
        y = np.random.randn(100)

        result = fit_predict(
            X,
            y,
            model_name="linear",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.REGRESSION,
            compute_roc=True,  # Should be ignored for regression
        )

        # ROC should not be computed for regression
        assert result.roc_diagnostics is None
        assert result.roc_artifacts == {}

    def test_roc_bootstrap_parameter(self, binary_classification_data):
        """Test that ROC bootstrap parameter is respected."""
        X, y = binary_classification_data

        result = fit_predict(
            X,
            y,
            model_name="logistic_regression",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.CLASSIFICATION,
            compute_roc=True,
            roc_n_bootstrap=500,
        )

        assert result.roc_diagnostics is not None
        # Bootstrap CIs should be available
        for class_key, metrics in result.roc_diagnostics.per_class.items():
            assert metrics.ci_lower is not None
            assert metrics.ci_upper is not None

    def test_roc_results_consistency(self, binary_classification_data):
        """Test consistency of ROC results across calls."""
        X, y = binary_classification_data

        # Run with ROC enabled
        result_with_roc = fit_predict(
            X,
            y,
            model_name="logistic_regression",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.CLASSIFICATION,
            compute_roc=True,
        )

        # Run without ROC
        result_without_roc = fit_predict(
            X,
            y,
            model_name="logistic_regression",
            scheme="kfold",
            outer_splits=3,
            seed=42,
            allow_random_cv=True,
            outcome_type=OutcomeType.CLASSIFICATION,
            compute_roc=False,
        )

        # Classification metrics should be identical
        assert np.array_equal(result_with_roc.y_true, result_without_roc.y_true)
        assert np.array_equal(result_with_roc.y_pred, result_without_roc.y_pred)
        assert np.allclose(result_with_roc.y_proba, result_without_roc.y_proba)
