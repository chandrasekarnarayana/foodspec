"""
Tests for leakage detection utilities.
"""

import numpy as np
import pytest

from foodspec.validation.leakage import FitLeakageGuard, mark_stateless


class DummyTransform:
    def __init__(self):
        self.fitted = False

    def fit(self, X: np.ndarray):
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


def test_leakage_guard_detects_full_data_fit():
    X_full = np.random.randn(100, 10)
    guard = FitLeakageGuard(n_full_samples=X_full.shape[0], stateless=False)
    with pytest.raises(AssertionError, match="Leakage detected"):
        guard.assert_fit(X_full)


def test_leakage_guard_allows_training_only_fit():
    X_full = np.random.randn(100, 10)
    X_train = X_full[:70]
    guard = FitLeakageGuard(n_full_samples=X_full.shape[0], stateless=False)
    # Should not raise
    guard.assert_fit(X_train)


def test_mark_stateless_bypasses_checks():
    X_full = np.random.randn(100, 10)
    t = DummyTransform()
    mark_stateless(t)
    guard = FitLeakageGuard(n_full_samples=X_full.shape[0], stateless=getattr(t, "stateless", False))
    # Should not raise due to stateless=True
    guard.assert_fit(X_full)
