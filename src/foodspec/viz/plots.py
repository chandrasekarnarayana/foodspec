from __future__ import annotations
"""High-level plotting helpers for common FoodSpec visualizations.

All functions return a Matplotlib Axes for further customization.
"""


from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_ax(ax=None):
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_spectra_overlay(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    ax=None,
    title: Optional[str] = None,
):
    """
    Overlay spectra for quick visual inspection.

    Parameters
    ----------
    spectra : np.ndarray
        Shape (n_samples, n_wavenumbers).
    wavenumbers : np.ndarray
        Shape (n_wavenumbers,).
    labels : sequence of str, optional
        Optional label per spectrum.
    ax : matplotlib Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib Axes
    """

    ax = _get_ax(ax)
    spectra = np.asarray(spectra)
    wavenumbers = np.asarray(wavenumbers)
    for i, y in enumerate(spectra):
        lab = labels[i] if labels is not None and i < len(labels) else None
        ax.plot(wavenumbers, y, alpha=0.7, label=lab)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    if labels is not None:
        ax.legend()
    if title:
        ax.set_title(title)
    return ax


def plot_mean_with_ci(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    group_labels: Sequence[str],
    ci: float = 95.0,
    ax=None,
):
    """
    Plot mean spectrum with confidence interval per group.

    Parameters
    ----------
    spectra : np.ndarray
        Shape (n_samples, n_wavenumbers).
    wavenumbers : np.ndarray
        Shape (n_wavenumbers,).
    group_labels : sequence of str
        Group label per spectrum.
    ci : float, optional
        Confidence interval percentile width, by default 95.
    ax : matplotlib Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib Axes
    """

    ax = _get_ax(ax)
    df = pd.DataFrame(spectra)
    df["__group"] = group_labels
    for name, grp in df.groupby("__group"):
        vals = grp.drop(columns="__group").to_numpy()
        mean = vals.mean(axis=0)
        lower = np.percentile(vals, (100 - ci) / 2, axis=0)
        upper = np.percentile(vals, 100 - (100 - ci) / 2, axis=0)
        ax.plot(wavenumbers, mean, label=name)
        ax.fill_between(wavenumbers, lower, upper, alpha=0.2)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend()
    return ax


def plot_pca_scores(
    scores: np.ndarray,
    labels: Optional[Sequence] = None,
    components: Tuple[int, int] = (1, 2),
    ax=None,
):
    """
    Scatter plot of PCA scores.

    Parameters
    ----------
    scores : np.ndarray
        Shape (n_samples, n_components).
    labels : sequence, optional
        Optional class/group labels.
    components : tuple, optional
        PCs to plot (1-based indexing), by default (1, 2).
    ax : matplotlib Axes, optional
        Axes to plot on.
    """

    ax = _get_ax(ax)
    c1, c2 = components
    x = scores[:, c1 - 1]
    y = scores[:, c2 - 1]
    if labels is not None:
        labels = np.asarray(labels)
        for lab in np.unique(labels):
            sel = labels == lab
            ax.scatter(x[sel], y[sel], label=str(lab), alpha=0.7)
        ax.legend()
    else:
        ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel(f"PC{c1}")
    ax.set_ylabel(f"PC{c2}")
    return ax


def plot_pca_loadings(
    loadings: np.ndarray,
    wavenumbers: np.ndarray,
    components: Tuple[int, int] = (1, 2),
    ax=None,
):
    """
    Plot PCA loadings for selected components.
    """

    ax = _get_ax(ax)
    c1, c2 = components
    ax.plot(wavenumbers, loadings[:, c1 - 1], label=f"PC{c1}")
    ax.plot(wavenumbers, loadings[:, c2 - 1], label=f"PC{c2}")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Loading")
    ax.legend()
    return ax


def plot_confusion_matrix(cm: np.ndarray, class_labels: Sequence[str], normalize: bool = False, ax=None):
    """
    Plot confusion matrix.
    """

    ax = _get_ax(ax)
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-12)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_correlation_heatmap(corr_matrix: np.ndarray, labels: Optional[Sequence[str]] = None, ax=None):
    """
    Plot correlation heatmap.
    """

    ax = _get_ax(ax)
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
    ax.set_title("Correlation matrix")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_regression_calibration(y_true: np.ndarray, y_pred: np.ndarray, ax=None):
    """
    Predicted vs true scatter with diagonal reference.
    """

    ax = _get_ax(ax)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ax.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", label="Ideal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.legend()
    return ax


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_value: Optional[float] = None, ax=None):
    """
    Plot ROC curve.
    """

    ax = _get_ax(ax)
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc_value:.3f})" if auc_value is not None else "ROC")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend()
    return ax


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, ax=None):
    """
    Plot precision-recall curve.
    """

    ax = _get_ax(ax)
    ax.plot(recall, precision, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, ax=None):
    """
    Plot residuals (y_true - y_pred) vs predicted.
    """

    ax = _get_ax(ax)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(0, color="k", linestyle="--", alpha=0.6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (true - pred)")
    return ax


def plot_calibration_with_ci(
    y_true,
    y_pred,
    ax=None,
    *,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: int | None = None,
    title: Optional[str] = None,
):
    """Plot predicted vs true with a bootstrap confidence band around a linear fit.

    Appropriate for calibration/regression to visualize bias and uncertainty.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ax = _get_ax(ax)

    ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="k", linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", label="1:1")

    # Fit simple linear regression for trend
    A = np.vstack([y_true, np.ones_like(y_true)]).T
    coef, intercept = np.linalg.lstsq(A, y_pred, rcond=None)[0]
    xx = np.linspace(*lims, 100)
    yy = coef * xx + intercept

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    band = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        coef_b, intercept_b = np.linalg.lstsq(A[idx], y_pred[idx], rcond=None)[0]
        band.append(coef_b * xx + intercept_b)
    band = np.asarray(band)
    lo = np.quantile(band, alpha / 2, axis=0)
    hi = np.quantile(band, 1 - alpha / 2, axis=0)

    ax.plot(xx, yy, color="C1", label="Fit")
    ax.fill_between(xx, lo, hi, color="C1", alpha=0.2, label=f"{int((1 - alpha) * 100)}% CI")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title or "Calibration with confidence band")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    return ax


def plot_bland_altman(
    y_true,
    y_pred,
    ax=None,
    *,
    ci_limits: bool = True,
    alpha: float = 0.05,
    title: Optional[str] = None,
):
    """Bland–Altman plot for agreement between predicted and reference values.

    Appropriate for method comparison/agreement studies in analytical chemistry.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_vals = (y_true + y_pred) / 2
    diff = y_pred - y_true
    bias = diff.mean()
    sd = diff.std(ddof=1)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    ax = _get_ax(ax)
    ax.scatter(mean_vals, diff, alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.axhline(bias, color="C1", label=f"Bias = {bias:.3f}")
    if ci_limits:
        ax.axhline(loa_low, color="C2", linestyle="--", label="Limits of agreement")
        ax.axhline(loa_high, color="C2", linestyle="--")
    ax.set_xlabel("Mean of methods")
    ax.set_ylabel("Predicted - True")
    ax.set_title(title or "Bland–Altman plot")
    ax.legend()
    return ax


__all__ = [
    "plot_spectra_overlay",
    "plot_mean_with_ci",
    "plot_correlation_heatmap",
    "plot_regression_calibration",
    "plot_residuals",
    "plot_calibration_with_ci",
    "plot_bland_altman",
]
