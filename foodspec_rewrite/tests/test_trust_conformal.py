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

from foodspec.models import LogisticRegressionClassifier
from foodspec.trust import MondrianConformalClassifier


def test_mondrian_conformal_hits_target_coverage() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(120, 5))
    y_train = np.array([0] * 40 + [1] * 40 + [2] * 40)
    rng.shuffle(y_train)
    X_cal = rng.normal(size=(60, 5))
    y_cal = np.array([0] * 20 + [1] * 20 + [2] * 20)
    rng.shuffle(y_cal)

    bins_cal = np.array(["early"] * 30 + ["late"] * 30)
    rng.shuffle(bins_cal)

    model = LogisticRegressionClassifier(random_state=0, max_iter=500, multi_class="multinomial")
    cp = MondrianConformalClassifier(model, target_coverage=0.8)
    cp.fit(X_train, y_train)
    cp.calibrate(X_cal, y_cal, bins=bins_cal)

    res_cal = cp.predict_sets(X_cal, bins=bins_cal, y_true=y_cal)

    assert res_cal.coverage is not None
    # Empirical coverage should meet or exceed target minus small tolerance
    assert res_cal.coverage >= 0.75
    assert set(res_cal.per_bin_coverage.keys()) == set(np.unique(bins_cal))


def test_unseen_bin_falls_back_to_global_threshold() -> None:
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(80, 4))
    y_train = np.array([0] * 40 + [1] * 40)
    rng.shuffle(y_train)
    X_cal = rng.normal(size=(40, 4))
    y_cal = np.array([0] * 20 + [1] * 20)
    rng.shuffle(y_cal)

    bins_cal = np.array(["instrument_a"] * 20 + ["instrument_b"] * 20)
    rng.shuffle(bins_cal)

    model = LogisticRegressionClassifier(random_state=0, max_iter=300)
    cp = MondrianConformalClassifier(model, target_coverage=0.85)
    cp.fit(X_train, y_train)
    cp.calibrate(X_cal, y_cal, bins=bins_cal)

    res = cp.predict_sets(X_cal[:2], bins=["new_site", bins_cal[1]], y_true=y_cal[:2])

    assert res.sample_thresholds[0] == pytest.approx(res.thresholds["global"])
    assert res.sample_thresholds[1] == pytest.approx(res.thresholds[str(bins_cal[1])])


def test_prediction_sets_sorted_by_probability() -> None:
    proba = np.array([[0.6, 0.3, 0.1]])
    cp = MondrianConformalClassifier(LogisticRegressionClassifier(), target_coverage=0.9)
    cp._thresholds = {"global": 0.7}
    cp._n_classes = 3

    res = cp.predict_sets_from_proba(proba)

    assert res.prediction_sets[0] == [0, 1]
    assert res.set_sizes[0] == 2
