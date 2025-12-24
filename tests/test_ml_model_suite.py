import pytest

from foodspec.chemometrics.models import (
    make_classifier,
    make_regressor,
    make_one_class_scanner,
)


def test_classification_factories_cover_suite():
    names = ["logreg", "svm_linear", "svm_rbf", "rf", "knn", "gboost", "mlp"]
    for name in names:
        model = make_classifier(name)
        assert hasattr(model, "fit")
    for opt in ("xgb", "lgbm"):
        try:
            model = make_classifier(opt)
            assert hasattr(model, "fit")
        except ImportError:
            pass
    with pytest.raises(ValueError):
        make_classifier("nope")


def test_regression_factories_cover_suite():
    names = ["ridge", "lasso", "elasticnet", "rf_reg", "mlp_reg"]
    for name in names:
        model = make_regressor(name)
        assert hasattr(model, "fit")
    for opt in ("xgb_reg", "lgbm_reg"):
        try:
            model = make_regressor(opt)
            assert hasattr(model, "fit")
        except ImportError:
            pass
    with pytest.raises(ValueError):
        make_regressor("nope")


def test_one_class_factories():
    oc = make_one_class_scanner("ocsvm")
    assert hasattr(oc, "fit")
    iso = make_one_class_scanner("isolation_forest")
    assert hasattr(iso, "fit")
    with pytest.raises(ValueError):
        make_one_class_scanner("unknown")
