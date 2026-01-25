"""Tests for trust component registration in ComponentRegistry."""

import numpy as np
import pandas as pd

from foodspec.core.registry import (
    ComponentRegistry,
    register_default_trust_components,
)
from foodspec.trust import (
    CombinedAbstainer,
    ConformalSizeAbstainer,
    MaxProbAbstainer,
    MondrianConformalClassifier,
    PlattCalibrator,
    IsotonicCalibrator,
)


class _DummyModel:
    def predict_proba(self, X):
        # Simple probability distribution for testing
        n_samples = len(X)
        return np.tile([0.6, 0.4], (n_samples, 1))

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _LinearModel:
    def __init__(self, coef):
        self.coef_ = np.asarray(coef)


def test_register_trust_components_available():
    registry = ComponentRegistry()
    register_default_trust_components(registry)

    assert {"platt", "isotonic"}.issubset(set(registry.available("calibrators")))
    assert {"mondrian"}.issubset(set(registry.available("conformal")))
    assert {"max_prob", "conformal_size", "combined"}.issubset(
        set(registry.available("abstain"))
    )
    assert {"coefficients", "permutation_importance"}.issubset(
        set(registry.available("interpretability"))
    )


def test_create_calibrators_and_conformal():
    registry = ComponentRegistry()
    register_default_trust_components(registry)

    platt = registry.create("calibrators", "platt")
    isotonic = registry.create("calibrators", "isotonic")
    conformal = registry.create("conformal", "mondrian", alpha=0.05, condition_key="batch")

    assert isinstance(platt, PlattCalibrator)
    assert isinstance(isotonic, IsotonicCalibrator)
    assert isinstance(conformal, MondrianConformalClassifier)
    assert conformal.alpha == 0.05
    assert conformal.condition_key == "batch"


def test_create_abstain_components():
    registry = ComponentRegistry()
    register_default_trust_components(registry)

    max_prob_rule = registry.create("abstain", "max_prob", threshold=0.7)
    size_rule = registry.create("abstain", "conformal_size", max_set_size=2)
    combined = registry.create("abstain", "combined", rules=[max_prob_rule, size_rule], mode="any")

    assert isinstance(max_prob_rule, MaxProbAbstainer)
    assert max_prob_rule.threshold == 0.7
    assert isinstance(size_rule, ConformalSizeAbstainer)
    assert size_rule.max_set_size == 2
    assert isinstance(combined, CombinedAbstainer)
    assert combined.mode == "any"
    assert len(combined.rules) == 2


def test_create_interpretability_components():
    registry = ComponentRegistry()
    register_default_trust_components(registry)

    coef_model = _LinearModel([[0.5, -0.25]])
    coef_df = registry.create(
        "interpretability",
        "coefficients",
        model=coef_model,
        feature_names=["f1", "f2"],
    )

    assert isinstance(coef_df, pd.DataFrame)
    assert set(["feature", "abs_coefficient"]).issubset(set(coef_df.columns))

    X = np.array([[1.0, 2.0], [0.5, -1.0]])
    y = np.array([0, 0])

    def accuracy(y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    importance_df = registry.create(
        "interpretability",
        "permutation_importance",
        model=_DummyModel(),
        X=X,
        y=y,
        metric_fn=accuracy,
        n_repeats=2,
        seed=0,
    )

    assert isinstance(importance_df, pd.DataFrame)
    assert {"feature", "importance_mean", "importance_std"}.issubset(
        set(importance_df.columns)
    )
