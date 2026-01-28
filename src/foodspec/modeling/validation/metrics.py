"""Validation utilities for chemometrics models."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, permutation_test_score

from foodspec.features.interpretation import find_chemical_meanings

__all__ = [
    "compute_classification_metrics",
    "compute_regression_metrics",
    "classification_metrics_bundle",
    "bootstrap_classification_ci",
    "cross_validate_pipeline",
    "permutation_test_score_wrapper",
    "compute_explained_variance",
    "compute_q2_rmsec_rmsep",
    "compute_vip_scores",
    "hotelling_t2_q_residuals",
    "permutation_pls_da",
    "make_pca_report",
    "vip_table_with_meanings",
    "classification_report_full",
    "regression_report_full",
    "confusion_matrix_table",
    "calibration_summary",
    "bootstrap_prediction_intervals",
    "split_conformal_regression",
    "reliability_diagram",
    "permutation_importance_wrapper",
    "compute_shap_values",
    "band_highlight_table",
]


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Compute common classification metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    average = "binary" if len(labels) == 2 else "weighted"
    pos_label = labels[0] if len(labels) == 2 else None

    results: dict[str, Any] = {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0, average=average, pos_label=pos_label),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0, average=average, pos_label=pos_label),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0, average=average, pos_label=pos_label),
    }
    if y_proba is not None and len(labels) == 2:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 2 and y_proba.shape[1] > 1:
            pos_scores = y_proba[:, 1]
        else:
            pos_scores = y_proba
        results["roc_auc"] = metrics.roc_auc_score(y_true, pos_scores)
        results["average_precision"] = metrics.average_precision_score(y_true, pos_scores)

    return pd.DataFrame([results])


def classification_report_full(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, average: str = "macro"
) -> pd.Series:
    """Full classification report with balanced accuracy, macro metrics, ROC-AUC, and Brier/calibration when probs supplied."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    np.unique(np.concatenate([y_true, y_pred]))
    avg = average
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    out: dict[str, float] = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        f"precision_{avg}": float(prec),
        f"recall_{avg}": float(rec),
        f"f1_{avg}": float(f1),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                roc = roc_auc_score(y_true, y_proba)
                brier = brier_score_loss(y_true, y_proba)
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                brier = brier_score_loss(y_true, y_proba[np.arange(len(y_proba)), np.argmax(y_proba, axis=1)])
            out["roc_auc"] = float(roc)
            out["brier_score"] = float(brier)
        except Exception:
            pass
    return pd.Series(out)


def classification_metrics_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    *,
    ece_bins: int = 15,
) -> dict:
    """Compute metrics, per-class metrics, and confusion matrices.

    Returns a dict with keys: metrics, per_class, confusion_matrix.
    """
    from foodspec.trust.reliability import expected_calibration_error

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    metrics_out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    ece = None
    roc_auc = None
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        try:
            ece = float(expected_calibration_error(y_true, y_proba, n_bins=ece_bins))
        except Exception:
            ece = None
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                roc_auc = float(roc_auc_score(y_true, y_proba))
            elif y_proba.shape[1] == 2:
                roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    if roc_auc is not None:
        metrics_out["roc_auc_ovr"] = roc_auc
    if ece is not None:
        metrics_out["ece"] = ece

    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {
        str(label): {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    confusion = {
        "labels": [str(label) for label in labels],
        "counts": cm_counts.tolist(),
        "normalized": cm_norm.tolist(),
    }

    return {
        "metrics": metrics_out,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def bootstrap_classification_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    *,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap confidence intervals for key classification metrics."""
    from foodspec.trust.reliability import expected_calibration_error

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float) if y_proba is not None else None

    metrics = {
        "accuracy": [],
        "f1_macro": [],
        "precision_macro": [],
        "recall_macro": [],
    }
    if y_proba is not None:
        metrics["roc_auc_ovr"] = []
        metrics["ece"] = []

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        metrics["accuracy"].append(float(accuracy_score(yt, yp)))
        metrics["f1_macro"].append(float(f1_score(yt, yp, average="macro", zero_division=0)))
        metrics["precision_macro"].append(float(precision_score(yt, yp, average="macro", zero_division=0)))
        metrics["recall_macro"].append(float(recall_score(yt, yp, average="macro", zero_division=0)))
        if y_proba is not None:
            yp_proba = y_proba[idx]
            try:
                if yp_proba.ndim == 1 or yp_proba.shape[1] == 1:
                    roc = roc_auc_score(yt, yp_proba)
                elif yp_proba.shape[1] == 2:
                    roc = roc_auc_score(yt, yp_proba[:, 1])
                else:
                    roc = roc_auc_score(yt, yp_proba, multi_class="ovr", average="macro")
                metrics["roc_auc_ovr"].append(float(roc))
            except Exception:
                pass
            try:
                metrics["ece"].append(float(expected_calibration_error(yt, yp_proba)))
            except Exception:
                pass

    ci: dict[str, dict] = {}
    for key, values in metrics.items():
        if not values:
            continue
        dist = np.asarray(values, dtype=float)
        ci[key] = {
            "mean": float(np.mean(dist)),
            "ci_low": float(np.quantile(dist, alpha / 2.0)),
            "ci_high": float(np.quantile(dist, 1.0 - alpha / 2.0)),
        }
    return ci


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """Compute regression metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    resid = y_pred - y_true
    std = float(np.std(resid))
    skew = float(np.mean(((resid - np.mean(resid)) / (std + 1e-12)) ** 3))
    kurtosis = float(np.mean(((resid - np.mean(resid)) / (std + 1e-12)) ** 4)) - 3.0
    return pd.Series(
        {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias, "residual_skew": skew, "residual_kurtosis": kurtosis}
    )


def regression_report_full(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """Extended regression report including bias and distribution stats."""

    return compute_regression_metrics(y_true, y_pred)


def cross_validate_pipeline(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int = 5,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Cross-validate a pipeline and return fold scores plus summary."""

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        return_train_score=False,
    )
    scores = cv_results["test_score"]
    rows = [{"fold": i + 1, "score": s} for i, s in enumerate(scores)]
    rows.append({"fold": "mean", "score": np.mean(scores)})
    rows.append({"fold": "std", "score": np.std(scores)})
    return pd.DataFrame(rows)


def confusion_matrix_table(y_true: np.ndarray, y_pred: np.ndarray, normalize: Optional[str] = None) -> pd.DataFrame:
    """Return confusion matrix as a DataFrame (optionally normalized)."""

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return pd.DataFrame(cm, index=labels, columns=labels)


def calibration_summary(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration curve and Brier score for binary classification."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba).ravel()
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_proba)
    return {"fraction_of_positives": frac_pos, "mean_predicted_value": mean_pred, "brier_score": float(brier)}


def permutation_test_score_wrapper(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = "accuracy",
    n_permutations: int = 100,
    random_state: Optional[int] = None,
):
    """Wrapper around sklearn's permutation_test_score."""

    score, perm_scores, pvalue = permutation_test_score(
        estimator,
        X,
        y,
        scoring=scoring,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return score, perm_scores, pvalue


def bootstrap_prediction_intervals(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 200,
    random_state: Optional[int] = None,
    X_eval: Optional[np.ndarray] = None,
):
    """Compute bootstrap prediction intervals for regression models.

    Returns lower/upper bounds per sample plus bootstrap mean predictions.
    """

    rng = np.random.default_rng(random_state)
    X_eval = X if X_eval is None else X_eval
    preds = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        model = clone(estimator)
        model.fit(Xb, yb)
        preds.append(model.predict(X_eval))
    pred_matrix = np.vstack(preds)
    lower = np.quantile(pred_matrix, alpha / 2, axis=0)
    upper = np.quantile(pred_matrix, 1 - alpha / 2, axis=0)
    mean_pred = np.mean(pred_matrix, axis=0)
    return {"lower": lower, "upper": upper, "mean": mean_pred}


def split_conformal_regression(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.05,
):
    """Split-conformal prediction for regression intervals."""

    model = clone(estimator)
    model.fit(X_train, y_train)
    calib_pred = model.predict(X_calib)
    residuals = np.abs(calib_pred - y_calib)
    q = np.quantile(residuals, 1 - alpha) * (1 + 1 / (len(residuals) + 1))
    test_pred = model.predict(X_test)
    return {"lower": test_pred - q, "upper": test_pred + q, "pred": test_pred}


def reliability_diagram(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Return reliability diagram bins (mean predicted vs. fraction positive)."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba).ravel()
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_of_positives": frac_pos})


def compute_explained_variance(pca: PCA) -> pd.Series:
    """Return explained variance and ratio for a fitted PCA."""

    return pd.Series(
        {
            "explained_variance_total": float(np.sum(pca.explained_variance_)),
            "explained_variance_ratio_mean": float(np.mean(pca.explained_variance_ratio_)),
        }
    )


def compute_q2_rmsec_rmsep(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.Series:
    """Compute Q², RMSEC (calibration), and RMSEP (prediction)."""

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_pred_cal = model.predict(X_train).ravel()
    y_pred_val = model.predict(X_test).ravel()

    sst = np.sum((y_test - np.mean(y_test)) ** 2) + 1e-12
    sse = np.sum((y_test - y_pred_val) ** 2)
    q2 = 1.0 - sse / sst

    rmsec = float(np.sqrt(metrics.mean_squared_error(y_train, y_pred_cal)))
    rmsep = float(np.sqrt(metrics.mean_squared_error(y_test, y_pred_val)))

    return pd.Series({"q2": q2, "rmsec": rmsec, "rmsep": rmsep})


def compute_vip_scores(pls_model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate Variable Importance in Projection (VIP) scores for PLSRegression."""

    if not hasattr(pls_model, "x_scores_"):
        raise ValueError("PLS model must be fitted before computing VIP.")
    T = pls_model.x_scores_
    W = pls_model.x_weights_
    Q = pls_model.y_loadings_.ravel()
    p, a = W.shape
    ssy = np.array([np.sum((T[:, comp] ** 2) * (Q[comp] ** 2)) for comp in range(a)])
    total_ssy = np.sum(ssy) + 1e-12
    vip = np.zeros(p)
    for j in range(p):
        weight = 0.0
        for comp in range(a):
            weight += ssy[comp] * (W[j, comp] ** 2) / (np.sum(W[:, comp] ** 2) + 1e-12)
        vip[j] = np.sqrt((p * weight) / total_ssy)
    return vip


def hotelling_t2_q_residuals(pca: PCA, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Hotelling T² and Q-residuals for PCA scores."""

    X = np.asarray(X, dtype=float)
    scores = pca.transform(X)
    eigvals = pca.explained_variance_
    t2 = np.sum((scores**2) / eigvals, axis=1)
    recon = pca.inverse_transform(scores)
    q_res = np.sum((X - recon) ** 2, axis=1)
    return t2, q_res


def permutation_pls_da(
    pipeline, X: np.ndarray, y: np.ndarray, n_permutations: int = 100, random_state: Optional[int] = None
):
    """Permutation test specialized for PLS-DA pipelines (accuracy scoring)."""

    return permutation_test_score_wrapper(
        pipeline, X, y, scoring="accuracy", n_permutations=n_permutations, random_state=random_state
    )


def permutation_importance_wrapper(
    estimator, X: np.ndarray, y: np.ndarray, n_repeats: int = 10, random_state: Optional[int] = None
):
    """Convenience wrapper for permutation_importance (classification or regression)."""

    result = permutation_importance(estimator, X, y, n_repeats=n_repeats, random_state=random_state)
    return result.importances_mean, result.importances_std


def make_pca_report(pca: PCA, scores: np.ndarray, loadings: np.ndarray) -> dict:
    """Assemble a lightweight PCA report object (scores/loadings + variance)."""

    return {
        "scores": scores,
        "loadings": loadings,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def vip_table_with_meanings(
    vip_scores: np.ndarray,
    wavenumbers: np.ndarray,
    top_n: int = 10,
    modality: str = "any",
    tolerance: float = 20.0,
) -> pd.DataFrame:
    """Return a table of VIP scores with optional chemistry interpretations."""

    if vip_scores.shape[0] != len(wavenumbers):
        raise ValueError("vip_scores and wavenumbers must align.")
    order = np.argsort(vip_scores)[::-1][:top_n]
    rows = []
    for idx in order:
        wn = float(wavenumbers[idx])
        meaning_matches = find_chemical_meanings(wn, modality=modality, tolerance=tolerance)
        rows.append(
            {
                "wavenumber": wn,
                "vip": float(vip_scores[idx]),
                "meaning": meaning_matches[0] if meaning_matches else "",
            }
        )
    return pd.DataFrame(rows)


def compute_shap_values(estimator, X: np.ndarray, nsamples: int = 100):
    """Compute SHAP values if shap is installed (optional dependency)."""

    try:
        import shap  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional
        raise ImportError("shap is required for compute_shap_values.") from exc

    explainer = shap.Explainer(estimator, X)
    return explainer(X[:nsamples])


def band_highlight_table(
    wavenumbers: np.ndarray,
    importance: np.ndarray,
    top_n: int = 10,
    modality: str = "any",
    tolerance: float = 20.0,
) -> pd.DataFrame:
    """Return top band/feature importances with chemical meaning for plotting highlights."""

    if importance.shape[0] != len(wavenumbers):
        raise ValueError("importance and wavenumbers must align.")
    order = np.argsort(importance)[::-1][:top_n]
    rows = []
    for idx in order:
        wn = float(wavenumbers[idx])
        meaning_matches = find_chemical_meanings(wn, modality=modality, tolerance=tolerance)
        rows.append(
            {
                "wavenumber": wn,
                "importance": float(importance[idx]),
                "meaning": meaning_matches[0] if meaning_matches else "",
            }
        )
    return pd.DataFrame(rows)
