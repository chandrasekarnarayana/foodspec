"""
Tests for trust integration inside evaluate_model_cv.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

from foodspec.trust.conformal import MondrianConformalClassifier, _mondrian_quantile
from foodspec.validation.evaluation import evaluate_model_cv


class RecordingCalibrator:
    """Identity calibrator that records calibration set sizes per fold."""

    calls = []

    def fit(self, y_true, proba):
        # Record size of calibration subset used in this fold
        RecordingCalibrator.calls.append(len(y_true))
        return self

    def predict(self, proba):
        return proba


class IdentityProbaModel:
    """Model that returns input rows as probabilities (assumes rows sum to 1)."""

    def fit(self, X, y):
        self.n_classes_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return np.asarray(X, dtype=float)


def test_calibrator_uses_only_calibration_subset_no_test_leakage():
    np.random.seed(123)
    n_samples = 60
    X = np.random.dirichlet([1, 1], size=n_samples)
    y = np.random.randint(0, 2, size=n_samples)

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    seed = 7
    RecordingCalibrator.calls = []

    evaluate_model_cv(
        X,
        y,
        model=IdentityProbaModel(),
        splitter=splitter,
        calibrator=RecordingCalibrator(),
        calibration_fraction=0.25,
        metrics=["accuracy"],
        seed=seed,
    )

    # Compute expected calibration sizes per fold using the same split logic
    expected_sizes = []
    for fold_id, (train_idx, _test_idx) in enumerate(splitter.split(X, y)):
        y_train = y[train_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + fold_id)
        _fit_local, cal_local = next(sss.split(np.zeros_like(y_train), y_train))
        expected_sizes.append(len(cal_local))

    assert RecordingCalibrator.calls == expected_sizes


def test_conformal_threshold_uses_calibration_only_and_saved_per_fold(tmp_path):
    # Deterministic data where probabilities are known
    X = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.52, 0.48],
            [0.48, 0.52],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1])

    splitter = KFold(n_splits=3, shuffle=False)
    cp = MondrianConformalClassifier(alpha=0.2)
    seed = 11

    evaluate_model_cv(
        X,
        y,
        model=IdentityProbaModel(),
        splitter=splitter,
        conformal_calibrator=cp,
        calibration_fraction=0.5,
        metrics=["accuracy"],
        seed=seed,
        trust_output_dir=tmp_path,
    )

    # Load thresholds saved for fold 0 and compare to expected value computed on calibration subset only
    conformal_path = tmp_path / "trust" / "conformal_sets_fold_0.csv"
    df = pd.read_csv(conformal_path)
    threshold_used = df["threshold"].iloc[0]

    # Recompute expected threshold manually
    train_idx, test_idx = next(splitter.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 0)
    _fit_local, cal_local = next(sss.split(np.zeros_like(y_train), y_train))
    proba_cal = X_train[cal_local]
    y_cal = y_train[cal_local]
    scores = 1.0 - proba_cal[np.arange(proba_cal.shape[0]), y_cal]
    expected_threshold = _mondrian_quantile(scores, 1.0 - cp.alpha)

    assert np.isclose(threshold_used, expected_threshold)


def test_trust_outputs_deterministic_with_seed(tmp_path):
    np.random.seed(99)
    n_samples = 40
    X = np.random.dirichlet([1, 1], size=n_samples)
    y = np.random.randint(0, 2, size=n_samples)
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=5)

    def run_once(outdir):
        evaluate_model_cv(
            X,
            y,
            model=IdentityProbaModel(),
            splitter=splitter,
            conformal_calibrator=MondrianConformalClassifier(alpha=0.1),
            calibration_fraction=0.3,
            metrics=["accuracy"],
            seed=3,
            trust_output_dir=outdir,
        )
        return pd.read_csv(outdir / "trust" / "coverage_overall.csv")

    cov1 = run_once(tmp_path / "run1")
    cov2 = run_once(tmp_path / "run2")

    pd.testing.assert_frame_equal(cov1, cov2)
