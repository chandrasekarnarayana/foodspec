import numpy as np

from foodspec.metrics import (
    compute_between_within_ratio,
    compute_between_within_stats,
    compute_classification_metrics,
    compute_embedding_silhouette,
    compute_pr_curve,
    compute_regression_metrics,
    compute_roc_curve,
)
from foodspec.viz import (
    plot_bland_altman,
    plot_calibration_with_ci,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_regression_calibration,
    plot_residuals,
    plot_roc_curve,
)


def test_classification_metrics_and_plots():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_scores = np.array([0.1, 0.6, 0.8, 0.9])
    res = compute_classification_metrics(y_true, y_pred, labels=[0, 1], y_scores=y_scores)
    assert "confusion_matrix" in res
    assert res["accuracy"] > 0
    cm = res["confusion_matrix"]
    ax = plot_confusion_matrix(cm, ["0", "1"])
    assert ax is not None
    if "roc_curve" in res:
        fpr, tpr = res["roc_curve"]
        ax = plot_roc_curve(fpr, tpr, res.get("auc"))
        assert ax is not None
        prec, rec, _ = compute_pr_curve(y_true, y_scores)
        ax = plot_pr_curve(prec, rec)
        assert ax is not None


def test_regression_metrics_and_plots():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    res = compute_regression_metrics(y_true, y_pred)
    assert res["rmse"] > 0
    ax = plot_regression_calibration(y_true, y_pred)
    assert ax is not None
    ax = plot_residuals(y_true, y_pred)
    assert ax is not None
    ax = plot_calibration_with_ci(y_true, y_pred)
    assert ax is not None
    ax = plot_bland_altman(y_true, y_pred)
    assert ax is not None


def test_roc_pr_helpers():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.3, 0.7, 0.9])
    fpr, tpr, _, auc_val = compute_roc_curve(y_true, y_scores)
    assert auc_val > 0.5
    prec, rec, _ = compute_pr_curve(y_true, y_scores)
    assert prec.size == rec.size


def test_embedding_structure_metrics():
    scores = np.array([[0, 0], [0.1, 0.1], [1, 1], [1.1, 1.1]])
    labels = np.array([0, 0, 1, 1])
    sil = compute_embedding_silhouette(scores, labels)
    assert sil >= 0
    ratio = compute_between_within_ratio(scores, labels)
    assert ratio > 5
    stats = compute_between_within_stats(scores, labels, n_permutations=20, random_state=0)
    assert "f_stat" in stats and stats["f_stat"] == stats["ratio"]
    assert np.isnan(stats["p_perm"]) or 0 <= stats["p_perm"] <= 1
