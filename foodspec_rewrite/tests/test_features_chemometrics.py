"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
"""

import numpy as np
import pytest

from foodspec.features import PCAFeatureExtractor


def test_pca_fit_transform_shapes_and_columns() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 10))
    pca = PCAFeatureExtractor(n_components=3, seed=0)

    Z_train, names = pca.fit_transform(X)
    assert names == ["pca_1", "pca_2", "pca_3"]
    assert Z_train.shape == (20, 3)
    evr = pca.explained_variance_ratio_
    assert evr.shape == (3,)
    assert 0.0 < evr.sum() <= 1.0


def test_pca_train_only_then_transform_holdout() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 8))
    X_train, X_test = X[:20], X[20:]

    pca = PCAFeatureExtractor(n_components=2, seed=42)
    Z_fit, names_fit = pca.fit_transform(X_train)
    Z_tr_again, names_tr_again = pca.transform(X_train)
    Z_te, names_te = pca.transform(X_test)

    # Fit-then-transform on train equals fit_transform on train
    assert np.allclose(Z_fit, Z_tr_again)
    # Transform works on test and has correct shape
    assert Z_te.shape == (10, 2)


def test_pca_input_validation() -> None:
    pca = PCAFeatureExtractor(n_components=2)

    with pytest.raises(ValueError):
        pca.fit(np.ones((10, 1)))  # n_components > n_features

    with pytest.raises(RuntimeError):
        pca.transform(np.ones((5, 2)))  # not fitted
