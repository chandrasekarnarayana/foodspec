#!/usr/bin/env python3
"""
Generate deterministic documentation figures for FoodSpec.

Run from repo root:
    python scripts/generate_docs_figures.py

Outputs:
- docs/assets/figures/*.png
- docs/assets/workflows/heating_quality_monitoring/heating_ratio_vs_time.png

All figures are generated from synthetic, deterministic data (seeded RNG) so
CI and local runs produce identical outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
FIG_OUT = ROOT / "docs" / "assets" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)
WF_OUT = ROOT / "docs" / "assets" / "workflows" / "heating_quality_monitoring"
WF_OUT.mkdir(parents=True, exist_ok=True)
AGING_OUT = ROOT / "docs" / "assets" / "workflows" / "aging"
AGING_OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
sns.set_theme(style="whitegrid")


def savefig(path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def fig_architecture():
    plt.figure(figsize=(6, 4))
    plt.axis("off")
    y = 0.9
    step = 0.12
    labels = [
        "Input (CSV/HDF5/vendor)",
        "IO layer (SpectralDataset / HyperspectralDataset)",
        "Preprocess + Harmonize",
        "RQ Engine",
        "Protocol Engine",
        "Bundles (report/figures/tables/models)",
        "GUI / CLI / Web",
    ]
    for lbl in labels:
        plt.text(0.5, y, lbl, ha="center", va="center")
        y -= step
    savefig(FIG_OUT / "architecture_flow.png")


def fig_confusion_and_pca():
    X, y = make_classification(
        n_samples=120,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        n_classes=4,
        n_clusters_per_class=1,
        weights=[0.25, 0.25, 0.25, 0.25],
        class_sep=2.0,
        random_state=42,
    )
    clf = RandomForestClassifier(n_estimators=80, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=5)
    classes = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=classes)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Oil authentication confusion matrix (synthetic)")
    savefig(FIG_OUT / "oil_confusion.png")

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=y, palette="tab10", s=40, edgecolor="none")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection (synthetic oils)")
    savefig(FIG_OUT / "oil_discriminative.png")


def fig_feature_panels():
    feats = [f"ratio_{i}" for i in range(1, 11)]
    rf_imp = np.linspace(0.25, 0.02, num=10) + RNG.normal(0, 0.01, 10)
    stability = np.linspace(0.5, 0.9, num=10) + RNG.normal(0, 0.01, 10)
    minimal_k = np.arange(2, 12)
    minimal_acc = 0.82 + 0.03 * np.exp(-0.3 * (minimal_k - 2)) + RNG.normal(0, 0.002, len(minimal_k))

    df_imp = pd.DataFrame({"feature": feats, "rf_importance": rf_imp})
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_imp, x="rf_importance", y="feature", orient="h")
    plt.title("Top discriminative ratios")
    savefig(FIG_OUT / "oil_discriminative.png")

    df_stab = pd.DataFrame({"feature": feats, "cv": stability})
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_stab, x="cv", y="feature", orient="h")
    plt.title("Top stable ratios")
    savefig(FIG_OUT / "oil_stability.png")

    plt.figure(figsize=(5, 4))
    plt.plot(minimal_k, minimal_acc, marker="o")
    plt.fill_between(minimal_k, minimal_acc - 0.01, minimal_acc + 0.01, alpha=0.2)
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.title("Minimal panel accuracy")
    savefig(FIG_OUT / "oil_minimal_panel.png")


def fig_oil_vs_chips():
    feats = [f"band_{i}" for i in range(1, 11)]
    effect = np.linspace(1.2, 0.3, num=10) + RNG.normal(0, 0.02, 10)
    df = pd.DataFrame({"feature": feats, "effect_size": effect})
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="effect_size", y="feature", orient="h")
    plt.title("Top matrix divergences")
    savefig(FIG_OUT / "oil_vs_chips_divergence.png")


def fig_cv_boxplot():
    scores = RNG.normal(loc=0.92, scale=0.02, size=30)
    plt.figure(figsize=(4, 4))
    sns.boxplot(y=scores)
    plt.ylabel("Balanced accuracy")
    plt.title("Cross-validation performance")
    savefig(FIG_OUT / "cv_boxplot.png")


def fig_heating_trend_and_workflow():
    time_h = np.arange(0, 20, 2)
    ratio = 0.05 * time_h + RNG.normal(0, 0.02, len(time_h))
    trend = np.poly1d(np.polyfit(time_h, ratio, 1))(time_h)

    plt.figure(figsize=(6, 4))
    plt.plot(time_h, ratio, "o", label="observed", color="#d62728")
    plt.plot(time_h, trend, "--", label="trend", color="#1f77b4")
    plt.xlabel("Time (hours)")
    plt.ylabel("Ratio (oxidation/preservation)")
    plt.title("Heating degradation trend")
    plt.legend()
    savefig(FIG_OUT / "heating_trend.png")

    # Workflow asset copy
    savefig(WF_OUT / "heating_ratio_vs_time.png")


def fig_hsi_and_roi():
    H, W, L = 40, 40, 400
    w = np.linspace(800, 3000, L)
    base = np.exp(-(((w - 1600) / 220) ** 2))
    cube = np.tile(base, (H, W, 1)) + RNG.normal(0, 0.01, (H, W, L))
    cube[15:28, 10:24, :] *= 0.6

    mean_img = cube.mean(axis=2)
    thresh = np.median(mean_img)
    labels = (mean_img < thresh).astype(int)

    plt.figure(figsize=(5, 4))
    sns.heatmap(labels, cmap="RdYlGn_r", cbar=False)
    plt.title("HSI label map (synthetic)")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    savefig(FIG_OUT / "hsi_label_map.png")

    healthy = cube[labels == 0].mean(axis=0)
    bruised = cube[labels == 1].mean(axis=0)
    plt.figure(figsize=(6, 4))
    plt.plot(w, healthy, label="Healthy", color="green")
    plt.plot(w, bruised, label="Bruised", color="red")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("ROI spectra")
    plt.legend()
    savefig(FIG_OUT / "roi_spectra.png")


def fig_aging_trend_and_workflow():
    days = np.arange(0, 181, 15)
    ratio = 0.003 * days + RNG.normal(0, 0.02, len(days)) + 0.35
    trend = np.poly1d(np.polyfit(days, ratio, 1))(days)

    plt.figure(figsize=(6, 4))
    plt.plot(days, ratio, "o", label="observed", color="#2ca02c")
    plt.plot(days, trend, "--", label="trend", color="#1f77b4")
    plt.xlabel("Storage Time (days)")
    plt.ylabel("Oxidation Ratio")
    plt.title("Aging degradation trend")
    plt.legend()
    savefig(AGING_OUT / "degradation_trajectories.png")

    # Shelf-life bar chart (synthetic t_star with CI)
    entities = [f"E{i}" for i in range(1, 6)]
    t_star = np.array([120, 150, 135, 110, 160], dtype=float)
    ci = np.array([15, 20, 18, 12, 22], dtype=float)
    plt.figure(figsize=(6, 4))
    plt.bar(entities, t_star, yerr=ci, capsize=4)
    plt.ylabel("Time to Threshold (days)")
    plt.title("Shelf-Life Estimates (synthetic)")
    savefig(AGING_OUT / "shelf_life_estimates.png")

    # Residuals vs time
    residuals = ratio - trend
    plt.figure(figsize=(6, 4))
    plt.axhline(0, color="gray", lw=1)
    plt.scatter(days, residuals, color="#ff7f0e")
    plt.xlabel("Storage Time (days)")
    plt.ylabel("Residual")
    plt.title("Residual diagnostics (linear fit)")
    savefig(AGING_OUT / "residual_plot.png")


def main():
    fig_architecture()
    fig_confusion_and_pca()
    fig_feature_panels()
    fig_oil_vs_chips()
    fig_cv_boxplot()
    fig_heating_trend_and_workflow()
    fig_hsi_and_roi()
    fig_aging_trend_and_workflow()


if __name__ == "__main__":  # pragma: no cover
    main()
