"""
Integration tests for EvaluationRunner and NestedCVRunner with stability selection.

These tests verify:
- Selection is integrated and produces marker_panel.json artifacts
- Selection is fit only on training folds (no leakage)
- Both standard and nested CV runners support optional selection
- Factory routing works with selection parameters
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from foodspec.features.selection import StabilitySelector
from foodspec.validation.evaluation import EvaluationRunner, create_evaluation_runner
from foodspec.validation.nested import NestedCVRunner


class SimpleSparseClassifier:
    """Simple L1-regularized-like classifier for testing."""

    def __init__(self, threshold=0.5, random_state=0):
        self.threshold = threshold
        self.random_state = random_state
        self.coef_ = None

    def fit(self, X, y):
        """Fit by computing correlation-based coefficients."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        # Simulate feature selection based on correlation magnitude
        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.standard_normal(n_features) * 2.0
        
        # Zero out weak features
        self.coef_[np.abs(self.coef_) < self.threshold] = 0.0
        return self

    def predict_proba(self, X):
        """Return dummy probabilities."""
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        proba = np.random.RandomState(42).uniform(0, 1, size=(n_samples, 2))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


@pytest.fixture
def sample_data():
    """Generate sample classification data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 20))
    y = rng.integers(0, 2, 100)
    return X, y


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestEvaluationRunnerWithSelection:
    """Test EvaluationRunner integration with stability selection."""

    def test_standard_cv_without_selection(self, sample_data, temp_output_dir):
        """Verify standard CV works without selection (backwards compat)."""
        X, y = sample_data
        estimator = SimpleSparseClassifier()
        runner = EvaluationRunner(
            estimator=estimator,
            n_splits=3,
            seed=42,
            output_dir=temp_output_dir,
        )
        result = runner.evaluate(X, y)
        
        # Check artifacts written
        assert (temp_output_dir / "predictions.csv").exists()
        assert (temp_output_dir / "metrics.csv").exists()
        # No marker panel without selector
        assert not (temp_output_dir / "marker_panel.json").exists()
        
        # Check result structure
        assert len(result.fold_predictions) > 0
        assert len(result.fold_metrics) == 3
        assert "accuracy" in result.bootstrap_ci

    def test_standard_cv_with_selection(self, sample_data, temp_output_dir):
        """Verify standard CV produces marker_panel.json with selection."""
        X, y = sample_data
        estimator = SimpleSparseClassifier()
        
        selector = StabilitySelector(
            estimator_factory=lambda: SimpleSparseClassifier(threshold=0.5),
            n_resamples=10,
            subsample_fraction=0.7,
            selection_threshold=0.3,
            random_state=42,
        )
        wavenumbers = np.linspace(1000, 2000, 20)
        
        runner = EvaluationRunner(
            estimator=estimator,
            n_splits=3,
            seed=42,
            output_dir=temp_output_dir,
            stability_selector=selector,
            x_wavenumbers=wavenumbers,
        )
        result = runner.evaluate(X, y)
        
        # Check all artifacts written
        assert (temp_output_dir / "predictions.csv").exists()
        assert (temp_output_dir / "metrics.csv").exists()
        assert (temp_output_dir / "marker_panel.json").exists()
        # Fold-level marker panels are saved per fold
        assert (temp_output_dir / "marker_panel_fold_0.json").exists()
        
        # Load and validate marker panel
        with open(temp_output_dir / "marker_panel.json", "r") as f:
            panel = json.load(f)
        
        assert "selected_indices" in panel
        assert "selection_frequencies" in panel
        assert "selected_wavenumbers" in panel
        assert "n_splits" in panel
        assert panel["n_splits"] == 3
        assert panel["subsample_fraction"] == 0.7
        assert panel["seed"] == 42
        
        # Verify wavenumbers match indices
        n_features = 20
        assert len(panel["selection_frequencies"]) == n_features
        for idx in panel["selected_indices"]:
            assert 0 <= idx < n_features

    def test_selection_without_wavenumbers(self, sample_data, temp_output_dir):
        """Verify marker panel works without wavenumbers."""
        X, y = sample_data
        estimator = SimpleSparseClassifier()
        
        selector = StabilitySelector(
            estimator_factory=lambda: SimpleSparseClassifier(),
            n_resamples=5,
            selection_threshold=0.3,
            random_state=42,
        )
        
        runner = EvaluationRunner(
            estimator=estimator,
            n_splits=3,
            seed=42,
            output_dir=temp_output_dir,
            stability_selector=selector,
            x_wavenumbers=None,  # No wavenumbers
        )
        result = runner.evaluate(X, y)
        
        # Marker panel created without wavenumbers
        assert (temp_output_dir / "marker_panel.json").exists()
        with open(temp_output_dir / "marker_panel.json", "r") as f:
            panel = json.load(f)
        
        assert "selected_indices" in panel
        assert "selection_frequencies" in panel
        # Wavenumbers should not be in panel
        assert "selected_wavenumbers" not in panel


class TestNestedCVRunnerWithSelection:
    """Test NestedCVRunner integration with stability selection."""

    def test_nested_cv_without_selection(self, sample_data, temp_output_dir):
        """Verify nested CV works without selection."""
        X, y = sample_data
        
        runner = NestedCVRunner(
            estimator_factory=lambda **p: SimpleSparseClassifier(),
            param_grid={"threshold": [0.3, 0.5]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=42,
            output_dir=temp_output_dir,
        )
        result = runner.evaluate(X, y)
        
        # Check artifacts
        assert (temp_output_dir / "predictions.csv").exists()
        assert (temp_output_dir / "metrics.csv").exists()
        assert (temp_output_dir / "hyperparameters_per_fold.csv").exists()
        # No marker panel without selector
        assert not (temp_output_dir / "marker_panel.json").exists()
        
        # Check result
        assert len(result.hyperparameters_per_fold) == 3
        assert len(result.fold_metrics) == 3

    def test_nested_cv_with_selection(self, sample_data, temp_output_dir):
        """Verify nested CV produces marker_panel.json with selection."""
        X, y = sample_data
        
        selector = StabilitySelector(
            estimator_factory=lambda: SimpleSparseClassifier(threshold=0.5),
            n_resamples=8,
            subsample_fraction=0.8,
            selection_threshold=0.4,
            random_state=42,
        )
        wavenumbers = np.linspace(1000, 2000, 20)
        
        runner = NestedCVRunner(
            estimator_factory=lambda **p: SimpleSparseClassifier(**p),
            param_grid={"threshold": [0.3, 0.5]},
            n_outer_splits=3,
            n_inner_splits=2,
            seed=42,
            output_dir=temp_output_dir,
            stability_selector=selector,
            x_wavenumbers=wavenumbers,
        )
        result = runner.evaluate(X, y)
        
        # Check all artifacts
        assert (temp_output_dir / "predictions.csv").exists()
        assert (temp_output_dir / "metrics.csv").exists()
        assert (temp_output_dir / "hyperparameters_per_fold.csv").exists()
        assert (temp_output_dir / "marker_panel.json").exists()
        
        # Load and validate marker panel
        with open(temp_output_dir / "marker_panel.json", "r") as f:
            panel = json.load(f)
        
        assert "selected_indices" in panel
        assert "selection_frequencies" in panel
        assert "selected_wavenumbers" in panel
        assert panel["n_splits"] == 3  # Outer splits
        assert panel["subsample_fraction"] == 0.8


class TestFactoryWithSelection:
    """Test create_evaluation_runner factory with selection."""

    def test_factory_standard_cv_with_selection(self, sample_data, temp_output_dir):
        """Verify factory creates standard runner with selection."""
        X, y = sample_data
        estimator = SimpleSparseClassifier()
        
        selector = StabilitySelector(
            estimator_factory=lambda: SimpleSparseClassifier(),
            n_resamples=5,
            selection_threshold=0.3,
            random_state=42,
        )
        wavenumbers = np.linspace(1000, 2000, 20)
        
        runner = create_evaluation_runner(
            estimator=estimator,
            nested=False,
            n_splits=3,
            seed=42,
            output_dir=temp_output_dir,
            stability_selector=selector,
            x_wavenumbers=wavenumbers,
        )
        
        result = runner.evaluate(X, y)
        assert isinstance(runner, EvaluationRunner)
        assert (temp_output_dir / "marker_panel.json").exists()

    def test_factory_nested_cv_with_selection(self, sample_data, temp_output_dir):
        """Verify factory creates nested runner with selection."""
        X, y = sample_data
        
        selector = StabilitySelector(
            estimator_factory=lambda: SimpleSparseClassifier(),
            n_resamples=5,
            selection_threshold=0.3,
            random_state=42,
        )
        wavenumbers = np.linspace(1000, 2000, 20)
        
        runner = create_evaluation_runner(
            estimator_factory=lambda **p: SimpleSparseClassifier(**p),
            param_grid={"threshold": [0.3, 0.5]},
            nested=True,
            n_splits=3,
            n_inner_splits=2,
            seed=42,
            output_dir=temp_output_dir,
            stability_selector=selector,
            x_wavenumbers=wavenumbers,
        )
        
        result = runner.evaluate(X, y)
        assert isinstance(runner, NestedCVRunner)
        assert (temp_output_dir / "marker_panel.json").exists()


class TestSelectionLeakageSafety:
    """Test that selection is fit only on training folds (leakage prevention)."""

    def test_selection_fit_on_training_only(self, sample_data, temp_output_dir):
        """Verify selection is fit on training folds, not test folds."""
        X, y = sample_data
        
        # Track which data selector is fit on
        fit_data_indices = []
        
        class TrackingStabilitySelector(StabilitySelector):
            """Selector that records which data it's fit on."""
            
            def fit(self, X_train, y_train):
                # Record that we're fitting only (would have access to sample indices if provided)
                fit_data_indices.append(("fit", X_train.shape[0]))
                return super().fit(X_train, y_train)
        
        selector = TrackingStabilitySelector(
            estimator_factory=lambda: SimpleSparseClassifier(),
            n_resamples=5,
            selection_threshold=0.3,
            random_state=42,
        )
        
        estimator = SimpleSparseClassifier()
        runner = EvaluationRunner(
            estimator=estimator,
            n_splits=3,
            seed=42,
            output_dir=temp_output_dir,
            stability_selector=selector,
        )
        result = runner.evaluate(X, y)
        
        # Selector should be fit 3 times (once per fold in run_stability_selection_cv)
        # Each time on training fold (not test fold)
        assert len(fit_data_indices) >= 3, f"Expected >= 3 fits, got {len(fit_data_indices)}"
        
        # All fits should be on training data (roughly 2/3 of samples per fold)
        for label, n_samples in fit_data_indices:
            assert label == "fit"
            # With 3 splits, training set ~67 samples, test set ~33
            assert 60 < n_samples < 75, f"Unexpected training set size: {n_samples}"


class TestSelectionDeterminism:
    """Test that selection with same seed produces same results."""

    def test_deterministic_marker_panel(self, sample_data, temp_output_dir):
        """Verify marker panel is deterministic with fixed seed."""
        X, y = sample_data
        
        results = []
        for i in range(2):
            tmpdir = temp_output_dir / f"run_{i}"
            tmpdir.mkdir()
            
            selector = StabilitySelector(
                estimator_factory=lambda: SimpleSparseClassifier(random_state=42),
                n_resamples=10,
                subsample_fraction=0.7,
                selection_threshold=0.3,
                random_state=42,  # Fixed seed
            )
            
            estimator = SimpleSparseClassifier(random_state=42)
            runner = EvaluationRunner(
                estimator=estimator,
                n_splits=3,
                seed=42,
                output_dir=tmpdir,
                stability_selector=selector,
            )
            result = runner.evaluate(X, y)
            
            with open(tmpdir / "marker_panel.json", "r") as f:
                panel = json.load(f)
            results.append(panel)
        
        # Same seed should produce same selection frequencies
        np.testing.assert_array_almost_equal(
            results[0]["selection_frequencies"],
            results[1]["selection_frequencies"],
            decimal=5,
        )
        assert results[0]["selected_indices"] == results[1]["selected_indices"]


__all__ = [
    "TestEvaluationRunnerWithSelection",
    "TestNestedCVRunnerWithSelection",
    "TestFactoryWithSelection",
    "TestSelectionLeakageSafety",
    "TestSelectionDeterminism",
]
