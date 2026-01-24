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

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from foodspec.models import LogisticRegressionClassifier
from foodspec.validation import EvaluationRunner, bootstrap_ci


def test_bootstrap_ci_deterministic_and_bounds() -> None:
    values = np.array([0.8, 0.85, 0.9, 0.75])
    ci1 = bootstrap_ci(values, n_bootstraps=100, ci=0.95, seed=42)
    ci2 = bootstrap_ci(values, n_bootstraps=100, ci=0.95, seed=42)

    # Deterministic with same seed
    assert np.allclose(ci1, ci2)
    # Bounds are in [min, max]
    assert values.min() <= ci1[0] <= ci1[1] <= values.max()


def test_evaluation_runner_with_artifact_saving(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.array([0, 1] * 20)

    clf = LogisticRegressionClassifier(random_state=0, max_iter=1000)
    runner = EvaluationRunner(clf, n_splits=5, seed=42, output_dir=str(tmp_path))

    result = runner.evaluate(X, y)

    # Check result structure
    assert len(result.fold_metrics) == 5
    assert len(result.fold_predictions) > 0
    assert "macro_f1" in result.bootstrap_ci
    assert "auroc" in result.bootstrap_ci
    assert "accuracy" in result.bootstrap_ci

    # Verify CIs are tuples with lower < upper
    for metric, (lower, upper) in result.bootstrap_ci.items():
        assert isinstance(lower, float) and isinstance(upper, float)
        assert lower <= upper

    # Check artifacts were saved
    metrics_path = tmp_path / "metrics.csv"
    preds_path = tmp_path / "predictions.csv"
    assert metrics_path.exists()
    assert preds_path.exists()

    # Verify CSV contents
    import csv
    with metrics_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert set(rows[0].keys()) >= {"fold_id", "accuracy", "macro_f1", "auroc"}


def test_evaluation_runner_deterministic_splits() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    y = np.random.randint(0, 2, 50)

    clf = LogisticRegressionClassifier(random_state=0, max_iter=1000)

    runner1 = EvaluationRunner(clf, n_splits=5, seed=123)
    runner2 = EvaluationRunner(clf, n_splits=5, seed=123)

    result1 = runner1.evaluate(X, y)
    result2 = runner2.evaluate(X, y)

    # Same seed -> same folds -> same predictions per fold
    f1_ci_1 = result1.bootstrap_ci.get("macro_f1")
    f1_ci_2 = result2.bootstrap_ci.get("macro_f1")
    if f1_ci_1 and f1_ci_2:
        assert np.allclose(f1_ci_1, f1_ci_2)


def test_evaluation_runner_without_artifacts_saves_nothing() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    y = np.array([0, 1] * 15)

    clf = LogisticRegressionClassifier(random_state=0, max_iter=1000)
    runner = EvaluationRunner(clf, n_splits=3, seed=0, output_dir=None)

    result = runner.evaluate(X, y)

    # Result is populated even without saving artifacts
    assert len(result.fold_metrics) == 3
    assert "accuracy" in result.bootstrap_ci
