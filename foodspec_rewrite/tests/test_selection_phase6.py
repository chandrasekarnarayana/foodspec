import numpy as np
import pytest

from foodspec.features.selection import StabilitySelector


def make_signal_data(seed=0, n_samples=200, n_features=60, signal_idx=(5, 17, 29)):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (
        X[:, signal_idx[0]] * 1.2
        + X[:, signal_idx[1]] * 0.9
        + X[:, signal_idx[2]] * 0.7
        + rng.normal(scale=0.5, size=n_samples)
        > 0
    ).astype(int)
    return X, y, list(signal_idx)


class MockL1Estimator:
    def __init__(self, C=1.0, random_state=0):
        self.C = C
        self.random_state = random_state
        self.coef_ = None

    def get_params(self):
        return {"C": self.C, "random_state": self.random_state}

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        corrs = np.abs(np.corrcoef(X.T, y)[-1, :-1])
        thresh = np.quantile(corrs, 0.85)
        coef = np.zeros(X.shape[1])
        mask = corrs >= thresh
        coef[mask] = rng.uniform(0.5, 1.0, size=mask.sum())
        self.coef_ = coef
        return self


class TestStabilitySelectionPhase6:
    def test_selector_recovers_signal_and_reports_panel(self):
        X, y, signal_idx = make_signal_data(seed=1)
        selector = StabilitySelector(
            estimator_factory=lambda: MockL1Estimator(C=1.0, random_state=7),
            n_resamples=30,
            subsample_fraction=0.7,
            selection_threshold=0.4,
            random_state=123,
        )
        selector.fit(X, y)
        panel = selector.get_marker_panel(
            x_wavenumbers=np.linspace(1000, 2000, X.shape[1]),
            feature_names=[f"f{i}" for i in range(X.shape[1])],
        )
        # Signal indices likely selected
        for i in signal_idx:
            assert i in panel["selected_indices"]
        # Deterministic with fixed seed
        selector2 = StabilitySelector(
            estimator_factory=lambda: MockL1Estimator(C=1.0, random_state=7),
            n_resamples=30,
            subsample_fraction=0.7,
            selection_threshold=0.4,
            random_state=123,
        )
        selector2.fit(X, y)
        assert np.allclose(selector.selection_frequencies_, selector2.selection_frequencies_)
        # Frequencies in [0,1] and sorted copy provided
        freqs = np.array(panel["selection_frequencies"]) 
        assert freqs.min() >= 0.0 and freqs.max() <= 1.0
        sorted_freqs = np.array(panel["selection_frequencies_sorted"]) 
        assert np.all(sorted_freqs[:-1] >= sorted_freqs[1:])
        # Panel contains threshold and model params
        assert panel["selection_threshold"] == pytest.approx(0.4)
        assert panel["model_params"]["random_state"] == 7
        # Transform reduces features
        X_reduced = selector.transform(X)
        assert X_reduced.shape[1] == len(panel["selected_indices"]) 

    def test_requires_fit_before_transform(self):
        X, y, _ = make_signal_data(seed=2)
        s = StabilitySelector(estimator_factory=lambda: MockL1Estimator())
        with pytest.raises(ValueError):
            _ = s.transform(X)
