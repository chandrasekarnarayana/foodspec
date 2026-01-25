"""End-to-end trust system test with deterministic artifacts (Phase 11)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.trust import (
    MondrianConformalClassifier,
    PlattCalibrator,
    extract_linear_coefficients,
    permutation_importance,
)
from foodspec.validation.evaluation import evaluate_model_cv


def _make_synthetic(seed: int = 123, n_per_class: int = 60, n_features: int = 6):
    """Create separable multiclass data with batch/stage metadata."""
    rng = np.random.default_rng(seed)
    centers = [
        np.linspace(-2.0, 0.0, n_features),
        np.linspace(1.5, 2.5, n_features),
        np.linspace(-1.0, 3.0, n_features),
    ]

    X_blocks = []
    y_blocks = []
    for cls, center in enumerate(centers):
        X_cls = rng.normal(loc=center, scale=0.35, size=(n_per_class, n_features))
        X_blocks.append(X_cls)
        y_blocks.append(np.full(n_per_class, cls, dtype=int))

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)

    n_samples = X.shape[0]
    batch_labels = np.tile(np.array(["batch_a", "batch_b", "batch_c"]), int(np.ceil(n_samples / 3)))[:n_samples]
    stage_labels = np.tile(np.array(["stage_pre", "stage_val"]), int(np.ceil(n_samples / 2)))[:n_samples]

    meta = pd.DataFrame({
        "batch": batch_labels,
        "stage": stage_labels,
    })
    # Duplicate batch into generic group key for GroupKFold compatibility
    meta["group"] = meta["batch"]
    return X, y, meta


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _run_trust_pipeline(run_dir: Path, seed: int = 123):
    X, y, meta = _make_synthetic(seed=seed)

    splitter = GroupKFold(n_splits=3)
    model = LogisticRegression(max_iter=400, random_state=seed)
    calibrator = PlattCalibrator()
    conformal = MondrianConformalClassifier(alpha=0.1, condition_key="stage", min_bin_size=1)

    # Execute evaluation with trust artifacts enabled
    evaluate_model_cv(
        X,
        y,
        model=model,
        splitter=splitter,
        calibrator=calibrator,
        calibration_fraction=0.25,
        conformal_calibrator=conformal,
        condition_key="stage",
        abstain_threshold=0.6,
        abstain_max_set_size=2,
        trust_output_dir=run_dir,
        seed=seed,
        meta=meta,
        metrics=["accuracy", "macro_f1"],
    )

    trust_dir = Path(run_dir) / "trust"

    # Calibrated probabilities should exist and rows sum to 1
    calibrated_files = sorted(trust_dir.glob("calibrated_test_proba_fold_*.csv"))
    assert calibrated_files, "Calibrated probability files not found"
    calibrated_df = pd.read_csv(calibrated_files[0])
    proba_cols = [c for c in calibrated_df.columns if c.startswith("proba_")]
    row_sums = calibrated_df[proba_cols].sum(axis=1)
    np.testing.assert_allclose(row_sums.values, np.ones_like(row_sums.values), atol=1e-6)

    # Conformal coverage overall should meet tolerance
    coverage_path = trust_dir / "coverage_overall.csv"
    assert coverage_path.exists(), "coverage_overall.csv missing"
    coverage_df = pd.read_csv(coverage_path)
    assert (coverage_df["coverage"] >= 0.85).all(), "Coverage below expected tolerance"

    # Conditional coverage table should have all stages present
    stages = set(meta["stage"].astype(str))
    assert stages.issubset(set(coverage_df["group"].astype(str))), "Missing stage rows in coverage table"

    # Abstention summary should expose rate and answered accuracy
    abstention_files = sorted(trust_dir.glob("abstention_fold_*.csv"))
    assert abstention_files, "Abstention summaries missing"
    abstention_df = pd.read_csv(abstention_files[0])
    assert {"abstain_rate", "accuracy_on_answered"}.issubset(abstention_df.columns)

    # Interpretability artifacts (coefficients + permutation importance)
    artifact_registry = ArtifactRegistry(run_dir)
    artifact_registry.ensure_layout()

    feature_names = [f"f{i}" for i in range(X.shape[1])]
    full_model = LogisticRegression(max_iter=400, random_state=seed)
    full_model.fit(X, y)

    coef_df = extract_linear_coefficients(full_model, feature_names)
    artifact_registry.write_csv(artifact_registry.coefficients_path, coef_df.to_dict(orient="records"))

    importance_df = permutation_importance(
        full_model,
        X,
        y,
        metric_fn=accuracy_score,
        n_repeats=3,
        seed=seed,
    )
    artifact_registry.write_csv(
        artifact_registry.permutation_importance_path,
        importance_df.to_dict(orient="records"),
    )

    assert artifact_registry.coefficients_path.exists()
    assert artifact_registry.permutation_importance_path.exists()

    # Collect hashes for deterministic checks
    key_files = [
        calibrated_files[0],
        coverage_path,
        abstention_files[0],
        artifact_registry.coefficients_path,
        artifact_registry.permutation_importance_path,
    ]
    return {path.name: _hash_file(path) for path in key_files}


def test_trust_end_to_end_deterministic():
    """Run trust pipeline twice and verify artifacts + determinism."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        hashes_run1 = _run_trust_pipeline(Path(tmpdir1), seed=123)
        hashes_run2 = _run_trust_pipeline(Path(tmpdir2), seed=123)

    assert hashes_run1 == hashes_run2, "Artifacts differ between deterministic runs"
