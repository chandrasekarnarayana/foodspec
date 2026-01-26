import numpy as np
import pytest
from sklearn.datasets import make_classification

from foodspec.core.errors import FoodSpecValidationError
from foodspec.modeling.api import fit_predict


def test_fit_predict_nested_schema():
    X, y = make_classification(n_samples=60, n_features=8, n_informative=6, random_state=0)
    result = fit_predict(
        X,
        y,
        model_name="logreg",
        scheme="nested",
        outer_splits=3,
        inner_splits=2,
        seed=0,
    )
    assert result.folds
    for fold in result.folds:
        assert "fold" in fold
        assert "train_idx" in fold
        assert "test_idx" in fold
        assert "metrics" in fold
        assert "confusion_matrix" in fold
    assert "accuracy" in result.metrics
    assert "f1_macro" in result.metrics


def test_random_cv_blocked_without_override():
    X, y = make_classification(n_samples=40, n_features=6, random_state=0)
    with pytest.raises(FoodSpecValidationError):
        fit_predict(X, y, model_name="logreg", scheme="random", allow_random_cv=False)
