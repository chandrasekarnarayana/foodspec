"""Unit tests for ROC/AUC diagnostics module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from foodspec.modeling.diagnostics import (
    PerClassRocMetrics,
    RocDiagnosticsResult,
    ThresholdResult,
    compute_auc_ci_bootstrap,
    compute_binary_roc_diagnostics,
    compute_multiclass_roc_diagnostics,
    compute_roc_diagnostics,
)


class TestBinaryROC:
    """Tests for binary classification ROC."""

    def test_binary_roc_separable_data(self):
        """Test ROC on perfectly separable data (AUC should be ~1.0)."""
        # Perfectly separable
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = compute_binary_roc_diagnostics(y_true, y_proba, random_seed=42)

        assert isinstance(result, RocDiagnosticsResult)
        assert len(result.per_class) == 1

        # Extract single class metrics
        metrics = list(result.per_class.values())[0]
        assert metrics.auc > 0.99
        assert metrics.n_positives == 3
        assert metrics.n_negatives == 3

    def test_binary_roc_ci(self):
        """Test CI computation for binary ROC."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = compute_binary_roc_diagnostics(y_true, y_proba, n_bootstrap=500, random_seed=42)
        metrics = list(result.per_class.values())[0]

        assert metrics.ci_lower is not None
        assert metrics.ci_upper is not None
        assert metrics.ci_lower <= metrics.auc
        assert metrics.auc <= metrics.ci_upper
        assert metrics.ci_lower >= 0
        assert metrics.ci_upper <= 1

    def test_binary_roc_youden_threshold(self):
        """Test Youden optimal threshold calculation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = compute_binary_roc_diagnostics(y_true, y_proba, random_seed=42)

        assert "youden" in result.optimal_thresholds
        thr = result.optimal_thresholds["youden"]
        assert isinstance(thr, ThresholdResult)
        assert 0 <= thr.sensitivity <= 1
        assert 0 <= thr.specificity <= 1
        assert thr.j_statistic == thr.sensitivity + thr.specificity - 1

    def test_binary_roc_random_classifier(self):
        """Test ROC on random classifier (AUC ~0.5)."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_proba = np.random.uniform(0, 1, 100)

        result = compute_binary_roc_diagnostics(y_true, y_proba, random_seed=42)
        metrics = list(result.per_class.values())[0]

        # For random data, AUC should be around 0.5 (±0.1 with high probability)
        assert 0.3 < metrics.auc < 0.7

    def test_binary_roc_with_sample_weights(self):
        """Test ROC with sample weights."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])
        weights = np.array([1, 1, 1, 2])  # Emphasize last positive

        result = compute_binary_roc_diagnostics(y_true, y_proba, sample_weight=weights, random_seed=42)
        metrics = list(result.per_class.values())[0]

        assert metrics.auc > 0.5


class TestMulticlassROC:
    """Tests for multiclass classification ROC."""

    def test_multiclass_roc_shapes(self):
        """Test multiclass ROC output shapes and structure."""
        X, y = make_classification(n_samples=150, n_features=20, n_informative=15, n_classes=3, n_clusters_per_class=1, random_state=42)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[:100], y[:100])
        y_proba = clf.predict_proba(X[100:])
        y_test = y[100:]

        result = compute_multiclass_roc_diagnostics(y_test, y_proba, n_bootstrap=100, random_seed=42)

        assert isinstance(result, RocDiagnosticsResult)
        assert len(result.per_class) == 3
        assert result.micro is not None
        assert result.macro_auc is not None

    def test_multiclass_roc_per_class_metrics(self):
        """Test per-class ROC metrics are valid."""
        X, y = make_classification(n_samples=120, n_features=15, n_informative=12, n_classes=3, n_clusters_per_class=1, random_state=42)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[:80], y[:80])
        y_proba = clf.predict_proba(X[80:])
        y_test = y[80:]

        result = compute_multiclass_roc_diagnostics(y_test, y_proba, random_seed=42)

        for label, metrics in result.per_class.items():
            assert isinstance(metrics, PerClassRocMetrics)
            assert 0 <= metrics.auc <= 1
            assert len(metrics.fpr) > 0
            assert len(metrics.tpr) == len(metrics.fpr)
            assert metrics.n_positives + metrics.n_negatives == len(y_test)

    def test_multiclass_roc_micro_macro(self):
        """Test micro/macro averaging."""
        X, y = make_classification(n_samples=120, n_features=15, n_informative=12, n_classes=3, n_clusters_per_class=1, random_state=42)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[:80], y[:80])
        y_proba = clf.predict_proba(X[80:])
        y_test = y[80:]

        result = compute_multiclass_roc_diagnostics(y_test, y_proba, random_seed=42)

        # Macro AUC should be average of per-class AUCs
        per_class_aucs = [metrics.auc for metrics in result.per_class.values()]
        expected_macro = np.mean(per_class_aucs)
        assert abs(result.macro_auc - expected_macro) < 1e-6

        # Micro AUC should be reasonable
        assert 0 <= result.micro.auc <= 1


class TestAutoTask:
    """Tests for automatic task detection."""

    def test_auto_binary_1d(self):
        """Auto-detect binary from 1D y_proba."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.2, 0.3, 0.7, 0.8])

        result = compute_roc_diagnostics(y_true, y_proba, task="auto", random_seed=42)
        assert len(result.per_class) == 1
        assert result.micro is None

    def test_auto_binary_2d(self):
        """Auto-detect binary from 2D y_proba."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]])

        result = compute_roc_diagnostics(y_true, y_proba, task="auto", random_seed=42)
        assert len(result.per_class) == 1

    def test_auto_multiclass(self):
        """Auto-detect multiclass."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

        result = compute_roc_diagnostics(y_true, y_proba, task="auto", random_seed=42)
        assert len(result.per_class) == 3
        assert result.micro is not None
        assert result.macro_auc is not None


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_ci_deterministic(self):
        """Test that bootstrap CI is deterministic with seed."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auc1, ci_lower1, ci_upper1 = compute_auc_ci_bootstrap(y_true, y_proba, n_bootstrap=100, random_seed=42)
        auc2, ci_lower2, ci_upper2 = compute_auc_ci_bootstrap(y_true, y_proba, n_bootstrap=100, random_seed=42)

        assert auc1 == auc2
        assert ci_lower1 == ci_lower2
        assert ci_upper1 == ci_upper2

    def test_bootstrap_ci_different_seeds(self):
        """Test that different seeds produce different CIs (with realistic data)."""
        # Use non-perfectly-separable data
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_proba = 0.5 + 0.3 * y_true + 0.2 * np.random.normal(0, 1, 100)
        y_proba = np.clip(y_proba, 0, 1)

        auc1, ci_lower1, ci_upper1 = compute_auc_ci_bootstrap(y_true, y_proba, n_bootstrap=100, random_seed=42)
        auc2, ci_lower2, ci_upper2 = compute_auc_ci_bootstrap(y_true, y_proba, n_bootstrap=100, random_seed=123)

        # With realistic data, intervals should be different (with high probability)
        # but may coincide if data is perfectly separable; allow some tolerance
        assert not (abs(ci_lower1 - ci_lower2) < 1e-10 and abs(ci_upper1 - ci_upper2) < 1e-10)

    def test_bootstrap_ci_bounds(self):
        """Test that CI bounds are valid."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        auc, ci_lower, ci_upper = compute_auc_ci_bootstrap(y_true, y_proba, n_bootstrap=100, random_seed=42)

        assert 0 <= ci_lower <= auc <= ci_upper <= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_mismatched_lengths(self):
        """Test error on mismatched y_true/y_proba lengths."""
        y_true = np.array([0, 1, 0])
        y_proba = np.array([0.2, 0.8])

        with pytest.raises(ValueError):
            compute_roc_diagnostics(y_true, y_proba)

    def test_single_class_error(self):
        """Test error on single class."""
        y_true = np.array([1, 1, 1, 1])
        y_proba = np.array([0.6, 0.7, 0.8, 0.9])

        with pytest.raises(ValueError):
            compute_multiclass_roc_diagnostics(y_true, y_proba.reshape(-1, 1))

    def test_tiny_sample(self):
        """Test with very small sample (2 samples)."""
        y_true = np.array([0, 1])
        y_proba = np.array([0.3, 0.7])

        result = compute_binary_roc_diagnostics(y_true, y_proba, random_seed=42)
        metrics = list(result.per_class.values())[0]

        assert metrics.auc == 1.0  # Perfect separation


class TestMetadata:
    """Tests for result metadata."""

    def test_metadata_binary(self):
        """Test metadata in binary result."""
        y_true = np.array([0, 1])
        y_proba = np.array([0.3, 0.7])

        result = compute_binary_roc_diagnostics(y_true, y_proba, n_bootstrap=100, random_seed=42)

        assert result.metadata["method"] == "binary_roc"
        assert result.metadata["n_bootstrap"] == 100
        assert result.metadata["random_seed"] == 42
        assert isinstance(result.metadata["warnings"], list)

    def test_metadata_multiclass(self):
        """Test metadata in multiclass result."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

        result = compute_multiclass_roc_diagnostics(y_true, y_proba, random_seed=42)

        assert result.metadata["method"] == "multiclass_ovr"
        assert result.metadata["random_seed"] == 42


class TestRealData:
    """Tests with realistic sklearn datasets."""

    def test_iris_like_binary(self):
        """Test on binary classification problem."""
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_classes=2, random_state=42)

        clf = LogisticRegression(random_state=42)
        clf.fit(X[:150], y[:150])
        y_proba = clf.predict_proba(X[150:])
        y_test = y[150:]

        result = compute_roc_diagnostics(y_test, y_proba, random_seed=42)

        # Should have good performance
        metrics = list(result.per_class.values())[0]
        assert metrics.auc > 0.6

    def test_digits_like_multiclass(self):
        """Test on multiclass problem."""
        X, y = make_classification(n_samples=300, n_features=30, n_informative=20, n_classes=5, n_clusters_per_class=2, random_state=42)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[:200], y[:200])
        y_proba = clf.predict_proba(X[200:])
        y_test = y[200:]

        result = compute_roc_diagnostics(y_test, y_proba, task="auto", random_seed=42)

        assert len(result.per_class) == 5
        assert result.macro_auc > 0.5
        assert result.micro.auc > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
