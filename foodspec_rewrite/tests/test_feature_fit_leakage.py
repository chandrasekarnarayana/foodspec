"""
Leakage-safety tests for feature extractor fitting inside CV folds.
"""

import numpy as np

from foodspec.features.base import FeatureExtractor
from foodspec.validation.evaluation import EvaluationRunner


class CountingExtractor(FeatureExtractor):
    fit_calls: list[int] = []
    transform_calls: list[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs):
        self.__class__.fit_calls.append(np.asarray(X).shape[0])
        return self

    def transform(self, X: np.ndarray, **kwargs):
        self.__class__.transform_calls.append(np.asarray(X).shape[0])
        X = np.asarray(X, dtype=float)
        return X[:, :1], ["f1"]


class SimpleEstimator:
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_features = X.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = np.asarray(X).shape[0]
        proba = np.tile(np.array([0.6, 0.4]), (n_samples, 1))
        return proba


def test_feature_extractor_fit_occurs_per_fold_only() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    y = np.array([0, 1] * 6)

    CountingExtractor.fit_calls = []
    CountingExtractor.transform_calls = []

    extractor = CountingExtractor()
    estimator = SimpleEstimator()
    runner = EvaluationRunner(
        estimator=estimator,
        feature_extractors=[extractor],
        n_splits=3,
        seed=42,
    )

    result = runner.evaluate(X, y)

    assert len(result.fold_metrics) == 3
    assert len(CountingExtractor.fit_calls) == runner.n_splits
    assert max(CountingExtractor.fit_calls) < X.shape[0]
    assert len(CountingExtractor.transform_calls) == runner.n_splits * 2  # train + val per fold