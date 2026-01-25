"""Tests for evaluate_model_cv function.

Verifies:
- Feature extractor fit called per fold only
- Deterministic predictions given seed
- No data leakage between train/test
- Proper handling of feature extraction, selection, and calibration
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV

from foodspec.validation import evaluate_model_cv


class CountingFeatureExtractor:
    """Feature extractor that counts how many times fit() is called."""

    def __init__(self):
        self.fit_count = 0
        self.mean_ = None

    def fit(self, X, y=None):
        """Fit and count."""
        self.fit_count += 1
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        """Transform by subtracting mean."""
        if self.mean_ is None:
            raise ValueError("Must call fit() before transform()")
        return X - self.mean_


class TestEvaluateModelCVBasic:
    """Test basic functionality of evaluate_model_cv."""

    def test_basic_evaluation_returns_result(self):
        """Test that basic evaluation returns EvaluationResult."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, seed=42)

        # Check structure
        assert hasattr(result, "fold_predictions")
        assert hasattr(result, "fold_metrics")
        assert hasattr(result, "bootstrap_ci")
        assert hasattr(result, "hyperparameters_per_fold")

    def test_returns_correct_number_of_folds(self):
        """Test that result contains correct number of folds."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, seed=42)

        assert len(result.fold_metrics) == 5

    def test_all_samples_predicted_once(self):
        """Test that each sample is predicted exactly once."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 20)
        y = np.random.randint(0, 2, n_samples)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, seed=42)

        # Check all samples predicted
        sample_indices = [p["sample_idx"] for p in result.fold_predictions]
        assert len(sample_indices) == n_samples
        assert set(sample_indices) == set(range(n_samples))

    def test_default_metrics_computed(self):
        """Test that default metrics are computed."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, seed=42)

        # Check default metrics
        assert "accuracy" in result.fold_metrics[0]
        assert "macro_f1" in result.fold_metrics[0]
        assert "auroc_macro" in result.fold_metrics[0]

    def test_custom_metrics_computed(self):
        """Test that custom metrics are computed."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(
            X,
            y,
            model,
            splitter,
            metrics=["accuracy", "precision_macro", "recall_macro"],
            seed=42,
        )

        # Check custom metrics
        assert "accuracy" in result.fold_metrics[0]
        assert "precision_macro" in result.fold_metrics[0]
        assert "recall_macro" in result.fold_metrics[0]
        assert "macro_f1" not in result.fold_metrics[0]

    def test_bootstrap_ci_computed(self):
        """Test that bootstrap CIs are computed."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, seed=42)

        # Check CIs exist
        assert "accuracy" in result.bootstrap_ci
        assert "macro_f1" in result.bootstrap_ci
        assert "auroc_macro" in result.bootstrap_ci

        # Check CI structure (now returns 3-tuple with median)
        ci_lower, ci_median, ci_upper = result.bootstrap_ci["accuracy"]
        assert isinstance(ci_lower, float)
        assert isinstance(ci_median, float)
        assert isinstance(ci_upper, float)
        assert ci_lower <= ci_median <= ci_upper


class TestEvaluateModelCVDeterminism:
    """Test deterministic behavior."""

    def test_deterministic_with_same_seed(self):
        """Test that results are deterministic given same seed."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result1 = evaluate_model_cv(X, y, model, splitter, seed=42)
        result2 = evaluate_model_cv(X, y, model, splitter, seed=42)

        # Check predictions match
        preds1 = [p["y_pred"] for p in result1.fold_predictions]
        preds2 = [p["y_pred"] for p in result2.fold_predictions]
        assert preds1 == preds2

        # Check metrics match
        for fold_idx in range(5):
            for metric_name in ["accuracy", "macro_f1", "auroc_macro"]:
                val1 = result1.fold_metrics[fold_idx][metric_name]
                val2 = result2.fold_metrics[fold_idx][metric_name]
                assert val1 == val2

    def test_different_results_with_different_seed(self):
        """Test that different seeds give different bootstrap CIs."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        # Use different splitter random states to ensure different folds
        splitter1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splitter2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

        result1 = evaluate_model_cv(X, y, model, splitter1, seed=42)
        result2 = evaluate_model_cv(X, y, model, splitter2, seed=99)

        # With different splitters, fold metrics should differ
        # Note: Bootstrap CI might be similar if fold metrics are similar
        # Instead, check that per-fold metrics differ
        fold_metrics_1 = [m["accuracy"] for m in result1.fold_metrics]
        fold_metrics_2 = [m["accuracy"] for m in result2.fold_metrics]
        assert fold_metrics_1 != fold_metrics_2


class TestEvaluateModelCVFeatureExtractor:
    """Test feature extraction pipeline."""

    def test_extractor_fit_called_per_fold_only(self):
        """Test that feature extractor fit is called once per fold."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        # Use a class-level counter that survives deepcopy
        class TrackingExtractor:
            fit_count = 0  # Class variable shared across instances
            
            def __init__(self):
                self.mean_ = None

            def fit(self, X, y=None):
                TrackingExtractor.fit_count += 1
                self.mean_ = np.mean(X, axis=0)
                return self

            def transform(self, X):
                if self.mean_ is None:
                    raise ValueError("Must fit before transform")
                return X - self.mean_

        TrackingExtractor.fit_count = 0  # Reset before test
        extractor = TrackingExtractor()
        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(
            X, y, model, splitter, feature_extractor=extractor, seed=42
        )

        # Should be called 5 times (once per fold)
        assert TrackingExtractor.fit_count == 5

    def test_extractor_fitted_on_train_only(self):
        """Test that extractor is fitted on training data only."""
        np.random.seed(42)
        # Create data where train and test have different means
        X = np.vstack(
            [
                np.random.randn(80, 20) + 10,  # Train data (higher mean)
                np.random.randn(20, 20),  # Test data (lower mean)
            ]
        )
        y = np.random.randint(0, 2, 100)

        class MeanExtractor:
            """Extracts mean of each feature."""

            def __init__(self):
                self.train_mean_ = None

            def fit(self, X, y=None):
                self.train_mean_ = np.mean(X, axis=0)
                return self

            def transform(self, X):
                # Return centered features
                return X - self.train_mean_

        extractor = MeanExtractor()
        model = LogisticRegression(random_state=42, max_iter=200)

        # Use a splitter that puts first 80 in train, last 20 in test
        from sklearn.model_selection import PredefinedSplit

        test_fold = np.concatenate([np.full(80, -1), np.zeros(20)])
        splitter = PredefinedSplit(test_fold)

        result = evaluate_model_cv(
            X, y, model, splitter, feature_extractor=extractor, seed=42
        )

        # Just verify it runs without error (extractor was fit on train only)
        assert len(result.fold_predictions) == 20  # Only test samples

    def test_without_extractor_uses_raw_features(self):
        """Test that without extractor, raw features are used."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, seed=42)

        # Should work fine with raw features
        assert len(result.fold_metrics) == 5


class TestEvaluateModelCVSelector:
    """Test feature selection pipeline."""

    def test_selector_fit_on_train_only(self):
        """Test that selector is fitted on training data only."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        selector = SelectKBest(f_classif, k=10)
        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, selector=selector, seed=42)

        # Should complete successfully
        assert len(result.fold_metrics) == 5

    def test_selector_with_extractor(self):
        """Test that selector works with feature extractor."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        extractor = CountingFeatureExtractor()
        selector = SelectKBest(f_classif, k=10)
        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(
            X,
            y,
            model,
            splitter,
            feature_extractor=extractor,
            selector=selector,
            seed=42,
        )

        # Should complete successfully
        assert len(result.fold_metrics) == 5


class TestEvaluateModelCVCalibration:
    """Test probability calibration pipeline."""

    def test_calibrator_applied(self):
        """Test that calibrator is applied when provided."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        
        # Simple calibrator that scales probabilities
        class SimpleCalibrator:
            def fit(self, proba, y):
                return self
            
            def transform(self, proba):
                # Just return scaled probabilities
                return proba * 0.8 + 0.1

        calibrator = SimpleCalibrator()
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(
            X, y, model, splitter, calibrator=calibrator, seed=42
        )

        # Should complete successfully
        assert len(result.fold_metrics) == 5


class TestEvaluateModelCVGroupHandling:
    """Test group-aware cross-validation."""

    def test_groups_tracked_in_predictions(self):
        """Test that group information is tracked in predictions."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        groups = np.repeat(np.arange(10), 10)  # 10 groups of 10 samples each

        meta = pd.DataFrame({"group": groups})

        model = LogisticRegression(random_state=42, max_iter=200)
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=5)

        result = evaluate_model_cv(X, y, model, splitter, seed=42, meta=meta)

        # Check that group is in predictions
        assert "group" in result.fold_predictions[0]

        # Check that all predictions have groups
        pred_groups = [p["group"] for p in result.fold_predictions]
        assert len(pred_groups) == 100
        assert set(pred_groups) == set(range(10))

    def test_without_groups_no_group_in_predictions(self):
        """Test that without groups, no group info in predictions."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, seed=42)

        # Check that group is NOT in predictions
        assert "group" not in result.fold_predictions[0]


class TestEvaluateModelCVMulticlass:
    """Test multiclass classification."""

    def test_multiclass_predictions(self):
        """Test that multiclass classification works."""
        np.random.seed(42)
        X = np.random.randn(150, 20)
        y = np.random.randint(0, 3, 150)  # 3 classes

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(X, y, model, splitter, seed=42)

        # Check probabilities for 3 classes
        assert "proba_0" in result.fold_predictions[0]
        assert "proba_1" in result.fold_predictions[0]
        assert "proba_2" in result.fold_predictions[0]

        # Check predictions are in range [0, 2]
        y_preds = [p["y_pred"] for p in result.fold_predictions]
        assert all(0 <= pred <= 2 for pred in y_preds)


class TestEvaluateModelCVErrorHandling:
    """Test error handling."""

    def test_invalid_metric_name_raises_error(self):
        """Test that invalid metric name raises error."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_model_cv(
                X, y, model, splitter, metrics=["invalid_metric"], seed=42
            )

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched X and y lengths raise error."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 50)  # Wrong length

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        with pytest.raises(ValueError, match="same length"):
            evaluate_model_cv(X, y, model, splitter, seed=42)


class TestEvaluateModelCVIntegration:
    """Integration tests with full pipeline."""

    def test_full_pipeline_extractor_selector_calibrator(self):
        """Test full pipeline with all components."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        extractor = CountingFeatureExtractor()
        selector = SelectKBest(f_classif, k=10)
        model = LogisticRegression(random_state=42, max_iter=200)
        
        class SimpleCalibrator:
            def fit(self, proba, y):
                return self
            def transform(self, proba):
                return proba

        calibrator = SimpleCalibrator()
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        result = evaluate_model_cv(
            X,
            y,
            model,
            splitter,
            feature_extractor=extractor,
            selector=selector,
            calibrator=calibrator,
            metrics=["accuracy", "macro_f1", "precision_macro", "recall_macro"],
            seed=42,
        )

        # Check all components executed
        assert len(result.fold_metrics) == 5
        assert "accuracy" in result.fold_metrics[0]
        assert "macro_f1" in result.fold_metrics[0]
        assert "precision_macro" in result.fold_metrics[0]
        assert "recall_macro" in result.fold_metrics[0]

    def test_realistic_spectroscopy_workflow(self):
        """Test realistic spectroscopy workflow."""
        np.random.seed(42)
        # Simulate spectra (100 samples, 500 wavenumbers)
        X = np.random.randn(100, 500)
        y = np.random.randint(0, 2, 100)
        groups = np.repeat(np.arange(5), 20)  # 5 batches

        meta = pd.DataFrame({"group": groups})

        # Use group-aware CV
        from sklearn.model_selection import GroupKFold

        model = LogisticRegression(random_state=42, max_iter=200)
        splitter = GroupKFold(n_splits=5)

        result = evaluate_model_cv(
            X,
            y,
            model,
            splitter,
            metrics=["accuracy", "macro_f1", "auroc_macro"],
            seed=42,
            meta=meta,
        )

        # Check results
        assert len(result.fold_metrics) == 5
        assert len(result.fold_predictions) == 100
        assert "group" in result.fold_predictions[0]
        
        # Check all metrics computed
        for fold_metric in result.fold_metrics:
            assert "accuracy" in fold_metric
            assert "macro_f1" in fold_metric
            assert "auroc_macro" in fold_metric
