import numpy as np
import pytest

from foodspec.chemometrics.models import make_classifier
from foodspec.chemometrics.validation import (
    compute_classification_metrics,
    compute_regression_metrics,
)


def test_make_classifier_supported_and_invalid():
    names = ["logreg", "svm_linear", "svm_rbf", "rf", "knn", "gboost"]
    for name in names:
        model = make_classifier(name)
        assert hasattr(model, "fit")
    with pytest.raises(ValueError):
        make_classifier("unknown")
    try:
        model = make_classifier("xgb")
        assert hasattr(model, "fit")
    except ImportError:
        pass
    try:
        model = make_classifier("lgbm")
        assert hasattr(model, "fit")
    except ImportError:
        pass


def test_validation_metrics_helpers():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    cls_df = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in cls_df.columns
    reg_series = compute_regression_metrics(y_true.astype(float), y_pred.astype(float))
    assert "rmse" in reg_series.index
