"""
Tests for stability selection marker panel.
"""

import numpy as np
import pytest

from foodspec.features.selection import StabilitySelector, run_stability_selection_cv


class MockSparseEstimator:
    """Estimator with L1-like behavior for testing (non-zero coefficients mark selection)."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.coef_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        # Simulate learning by correlating with y: features with higher correlation get non-zero coeff
        corrs = np.abs(np.corrcoef(X.T, y)[-1, :-1])
        # Threshold to select top ~10% features
        thresh = np.quantile(corrs, 0.9)
        coef = np.zeros(X.shape[1])
        coef[corrs >= thresh] = rng.uniform(0.5, 1.0, size=(corrs >= thresh).sum())
        self.coef_ = coef
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Not used in selection; stub for interface
        rng = np.random.default_rng(self.random_state)
        return rng.dirichlet(np.ones(2), size=X.shape[0])


def test_stability_selector_basic_selection():
    """StabilitySelector selects high-signal features reliably."""
    rng = np.random.default_rng(123)
    n_samples, n_features = 200, 100
    X = rng.normal(size=(n_samples, n_features))
    # Inject signal at known indices
    signal_idx = [10, 20, 30]
    y = (X[:, signal_idx[0]] + X[:, signal_idx[1]] * 0.8 + X[:, signal_idx[2]] * 0.6 + rng.normal(scale=0.3, size=n_samples) > 0).astype(int)

    selector = StabilitySelector(
        estimator_factory=lambda: MockSparseEstimator(random_state=42),
        n_resamples=20,
        subsample_fraction=0.7,
        selection_threshold=0.4,
        random_state=999,
    )

    selector.fit(X, y)
    panel = selector.get_marker_panel(x_wavenumbers=np.linspace(400, 500, n_features))

    # Ensure signal indices are selected
    for i in signal_idx:
        assert i in panel["selected_indices"]
    # Frequencies within [0,1]
    freqs = np.array(panel["selection_frequencies"]) 
    assert freqs.min() >= 0.0 and freqs.max() <= 1.0


def test_run_stability_selection_cv_aggregates_and_saves(tmp_path):
    """run_stability_selection_cv aggregates across folds and writes marker_panel.json."""
    rng = np.random.default_rng(321)
    n_samples, n_features = 120, 80
    X = rng.normal(size=(n_samples, n_features))
    signal_idx = [5, 7, 13]
    y = (X[:, signal_idx[0]] + X[:, signal_idx[1]] * 0.8 + X[:, signal_idx[2]] * 0.6 + rng.normal(scale=0.3, size=n_samples) > 0).astype(int)
    x = np.linspace(400, 500, n_features)

    selector = StabilitySelector(
        estimator_factory=lambda: MockSparseEstimator(random_state=7),
        n_resamples=10,
        subsample_fraction=0.7,
        selection_threshold=0.3,
        random_state=777,
    )

    outdir = tmp_path / "selection_run"
    panel = run_stability_selection_cv(
        estimator_factory=lambda **p: MockSparseEstimator(random_state=7),
        selector=selector,
        X=X,
        y=y,
        x_wavenumbers=x,
        n_splits=3,
        seed=42,
        output_dir=outdir,
    )

    # Artifact exists
    assert (outdir / "marker_panel.json").exists()
    # Selected indices non-empty and include signal idx likely
    assert len(panel["selected_indices"]) > 0
    assert any(i in panel["selected_indices"] for i in signal_idx)


def test_selector_requires_fit_before_transform():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 10))
    y = rng.integers(0, 2, size=30)
    s = StabilitySelector(estimator_factory=lambda: MockSparseEstimator())
    with pytest.raises(ValueError):
        s.transform(X)


def test_selector_invalid_inputs():
    s = StabilitySelector(estimator_factory=lambda: MockSparseEstimator(), subsample_fraction=0.0)
    X = np.zeros((10, 5))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        s.fit(X, y)
