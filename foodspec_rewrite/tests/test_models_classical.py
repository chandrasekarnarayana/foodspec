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

import os
from pathlib import Path

import numpy as np
import pytest

from foodspec.models import LogisticRegressionClassifier


def test_logreg_fit_predict_proba_and_shapes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n, d = 60, 12
    X_pos = rng.normal(loc=1.0, scale=0.5, size=(n // 2, d))
    X_neg = rng.normal(loc=-1.0, scale=0.5, size=(n // 2, d))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [0] * (n // 2))

    clf = LogisticRegressionClassifier(random_state=0, max_iter=1000)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    preds = clf.predict(X)
    assert preds.shape == (n,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Save/load round-trip
    model_path = tmp_path / "lr_model.joblib"
    clf.save(model_path)
    assert model_path.exists()

    loaded = LogisticRegressionClassifier.load(model_path)
    proba_loaded = loaded.predict_proba(X)
    assert np.allclose(proba_loaded, proba)


def test_logreg_not_fitted_errors() -> None:
    clf = LogisticRegressionClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(np.zeros((2, 3)))
    with pytest.raises(RuntimeError):
        clf.predict_proba(np.zeros((2, 3)))

    with pytest.raises(TypeError):
        LogisticRegressionClassifier(n_components=2)  # type: ignore[arg-type]
