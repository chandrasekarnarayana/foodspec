"""Reproduce the core analyses from the MethodsX protocol."""

from __future__ import annotations

from os import PathLike
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.chemometrics.pca import run_pca
from foodspec.data.public import (
    load_public_evoo_sunflower_raman,
    load_public_mendeley_oils,
)
from foodspec.reporting import (
    create_run_dir,
    save_figure,
    summarize_metrics_for_markdown,
    write_json,
    write_markdown_report,
)
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.viz.pca import plot_pca_scores

__all__ = ["run_methodsx_reproduction"]


def run_methodsx_reproduction(output_dir: PathLike, random_state: int = 42) -> Dict:
    """
    Run the core analyses used in the MethodsX protocol article and save artifacts.

    Steps
    -----
    1) Oil-type classification on a public dataset (Raman/FTIR).
    2) PCA visualization of the same dataset.
    3) Mixture analysis on EVOO–sunflower data via regression/NNLS-like fit.

    Returns
    -------
    dict
        High-level metrics including accuracy/F1 for classification and
        R²/RMSE for mixture regression. A ``run_dir`` key points to the
        folder containing artifacts.
    """

    run_dir = create_run_dir(output_dir, "methodsx")
    metrics: Dict[str, float] = {}

    # Oil classification (classic ML only)
    oil_ds = load_public_mendeley_oils()
    oil_result = run_oil_authentication_workflow(oil_ds, label_column="oil_type", cv_splits=3)
    if "accuracy" in oil_result.cv_metrics.columns:
        metrics["oil_accuracy"] = float(oil_result.cv_metrics["accuracy"].mean())
    if "f1" in oil_result.cv_metrics.columns:
        metrics["oil_f1"] = float(oil_result.cv_metrics["f1"].mean())

    # Confusion matrix plot
    fig_cm, ax_cm = plt.subplots()
    plot_confusion_matrix(oil_result.confusion_matrix, class_names=oil_result.class_labels, ax=ax_cm)
    save_figure(run_dir, "oil_confusion_matrix", fig_cm)
    plt.close(fig_cm)

    # PCA visualization
    _, pca_res = run_pca(oil_ds.x, n_components=2)
    fig_pca, ax_pca = plt.subplots()
    plot_pca_scores(pca_res.scores, labels=oil_ds.metadata.get("oil_type"), ax=ax_pca)
    ax_pca.set_title("PCA scores (public oil dataset)")
    save_figure(run_dir, "oil_pca_scores", fig_pca)
    plt.close(fig_pca)

    # Mixture regression (EVOO–sunflower)
    mix_ds = load_public_evoo_sunflower_raman()
    y = mix_ds.metadata["mixture_fraction_evoo"].to_numpy()
    mask = ~np.isnan(y)
    X = mix_ds.x[mask]
    y = y[mask]
    # If values look like percentages, scale to 0-1
    if np.nanmax(y) > 1:
        y = y / 100.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=None
    )
    reg = Pipeline([("scaler", StandardScaler()), ("linreg", LinearRegression())])
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    metrics["mixture_r2"] = float(r2_score(y_test, y_pred))
    metrics["mixture_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Save metrics and report
    metrics["run_dir"] = str(run_dir)
    write_json(run_dir / "metrics.json", metrics)
    sections = {
        "Overview": (
            "This run executes the core MethodsX protocol reproduction: oil-type "
            "classification, PCA visualization, and EVOO–sunflower mixture regression."
        ),
        "Key metrics": summarize_metrics_for_markdown(metrics),
        "Artifacts": (
            "- oil_confusion_matrix.png\n"
            "- oil_pca_scores.png\n"
            "- metrics.json"
        ),
        "Assumptions": (
            "Public datasets must be downloaded locally for the loaders to succeed."
        ),
    }
    write_markdown_report(run_dir / "report.md", title="MethodsX Protocol Reproduction", sections=sections)
    return metrics
