import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from foodspec.chemometrics.validation import (
    classification_report_full,
    regression_report_full,
    confusion_matrix_table,
    calibration_summary,
    bootstrap_prediction_intervals,
    split_conformal_regression,
    reliability_diagram,
    permutation_importance_wrapper,
    band_highlight_table,
)


def test_classification_metrics_and_calibration():
    X, y = make_classification(n_samples=80, n_features=5, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    preds = clf.predict(Xte)
    report = classification_report_full(yte, preds, proba)
    assert report["accuracy"] > 0.6
    cm = confusion_matrix_table(yte, preds)
    assert cm.shape[0] == cm.shape[1]
    calib = calibration_summary(yte, proba, n_bins=5)
    assert "brier_score" in calib
    rel = reliability_diagram(yte, proba, n_bins=5)
    assert set(rel.columns) == {"mean_predicted", "fraction_of_positives"}


def test_regression_metrics_uncertainty_and_conformal():
    X, y = make_regression(n_samples=100, n_features=6, noise=5.0, random_state=2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=3)
    lr = LinearRegression().fit(Xtr, ytr)
    preds = lr.predict(Xte)
    reg = regression_report_full(yte, preds)
    assert "bias" in reg
    boot = bootstrap_prediction_intervals(lr, Xtr, ytr, alpha=0.1, n_bootstrap=30, random_state=0, X_eval=Xte)
    assert boot["lower"].shape == preds.shape
    conformal = split_conformal_regression(lr, Xtr, ytr, Xte[:20], yte[:20], Xte[:10], alpha=0.1)
    assert conformal["lower"].shape[0] == 10


def test_permutation_importance_and_band_highlight():
    X, y = make_classification(n_samples=60, n_features=8, random_state=4)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    mean_imp, std_imp = permutation_importance_wrapper(clf, X, y, n_repeats=3, random_state=0)
    assert mean_imp.shape[0] == X.shape[1]
    wn = np.linspace(1000, 1100, num=X.shape[1])
    table = band_highlight_table(wn, mean_imp, top_n=5, modality="raman")
    assert len(table) == 5


@pytest.mark.skipif("shap" not in globals(), reason="shap optional; not imported here")
def test_placeholder_shap_import():
    # This placeholder ensures optional dependency is acknowledged.
    assert True
