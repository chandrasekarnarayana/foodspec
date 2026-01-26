from __future__ import annotations

import numpy as np

from foodspec.engine.preprocessing.harmonization import (
    apply_direct_standardization,
    apply_piecewise_direct_standardization,
    apply_subspace_alignment,
)


def test_direct_standardization_shapes():
    rng = np.random.default_rng(0)
    X_source = rng.normal(size=(12, 40))
    X_target = rng.normal(size=(12, 40))
    X_prod = rng.normal(size=(5, 40))
    corrected, metrics = apply_direct_standardization(X_source, X_target, X_prod, alpha=0.5)
    assert corrected.shape == X_prod.shape
    assert "reconstruction_rmse" in metrics


def test_piecewise_standardization_shapes():
    rng = np.random.default_rng(1)
    X_source = rng.normal(size=(12, 40))
    X_target = rng.normal(size=(12, 40))
    X_prod = rng.normal(size=(5, 40))
    corrected, metrics = apply_piecewise_direct_standardization(X_source, X_target, X_prod, window_size=7)
    assert corrected.shape == X_prod.shape
    assert "window_size" in metrics


def test_subspace_alignment_shapes():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(20, 50))
    target = rng.normal(size=(10, 50))
    aligned, metrics = apply_subspace_alignment(source, target, n_components=5)
    assert aligned.shape == target.shape
    assert "alignment_shift_magnitude" in metrics
