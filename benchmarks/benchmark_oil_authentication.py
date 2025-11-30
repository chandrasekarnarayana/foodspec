# %% [markdown]
# # Benchmark: Edible Oil Authentication (foodspec vs. manual sklearn)
#
# This Jupytext-friendly script benchmarks the foodspec workflow against a manual
# scikit-learn implementation. Goals:
# - Compare ease-of-use (rough line counts)
# - Compare accuracy/F1 on a synthetic edible oils dataset
# - Visualize cross-validation metrics and confusion matrices

# %%
from __future__ import annotations

import inspect
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.data.loader import load_example_oils
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother


# %% [markdown]
# ## Helper: manual preprocessing and features (no foodspec abstractions)

# %%
def manual_preprocess_and_features(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    expected_peaks: List[float],
    tolerance: float = 8.0,
) -> np.ndarray:
    """Manual preprocessing: polynomial baseline, Savitzky-Golay smoothing, vector norm,
    and simple peak height extraction within a tolerance window."""

    # Baseline (poly-2) subtraction
    x_axis = np.linspace(0, 1, X.shape[1])
    X_bl = np.zeros_like(X)
    for i, y in enumerate(X):
        coefs = np.polyfit(x_axis, y, deg=2)
        X_bl[i] = y - np.polyval(coefs, x_axis)

    # Smooth
    X_sm = savgol_filter(X_bl, window_length=9, polyorder=3, axis=1)

    # Vector norm
    norms = np.linalg.norm(X_sm, axis=1, keepdims=True)
    norms = np.maximum(norms, np.finfo(float).eps)
    X_norm = X_sm / norms

    # Peak heights
    feats = []
    for y in X_norm:
        row_feats = []
        for center in expected_peaks:
            mask = (wavenumbers >= center - tolerance) & (wavenumbers <= center + tolerance)
            if not np.any(mask):
                row_feats.append(np.nan)
            else:
                row_feats.append(float(y[mask].max()))
        feats.append(row_feats)
    return np.asarray(feats)


# %% [markdown]
# ## foodspec workflow

# %%
def run_foodspec_workflow():
    ds = load_example_oils()
    result = run_oil_authentication_workflow(ds, label_column="oil_type", classifier_name="rf", cv_splits=5)
    print("foodspec CV metrics:")
    print(result.cv_metrics)
    print("\nConfusion matrix:")
    print(result.confusion_matrix)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(result.confusion_matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(result.class_labels, rotation=45, ha="right")
    ax.set_yticklabels(result.class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
    return result


# %% [markdown]
# ## Manual sklearn workflow (baseline/smoothing/normalization + peak heights + RF)

# %%
def run_manual_workflow():
    ds = load_example_oils()
    X = ds.x
    wn = ds.wavenumbers
    y = ds.metadata["oil_type"].to_numpy()
    expected_peaks = [1655.0, 1742.0, 1450.0]

    feats = manual_preprocess_and_features(X, wn, expected_peaks=expected_peaks)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    y_pred = cross_val_predict(clf, feats, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, pos_label="olive")
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))

    print("Manual sklearn CV (approximate, via cross_val_predict):")
    print(f"Accuracy: {acc:.3f}, F1 (olive vs rest): {f1:.3f}")
    print("Confusion matrix:\n", cm)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(np.unique(y), rotation=45, ha="right")
    ax.set_yticklabels(np.unique(y))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()

    return {"acc": acc, "f1": f1, "cm": cm}


# %% [markdown]
# ## Run both workflows and compare

# %%
if __name__ == "__main__":
    print("=== Running foodspec workflow ===")
    food_result = run_foodspec_workflow()

    print("\n=== Running manual sklearn workflow ===")
    manual_result = run_manual_workflow()

    # Rough code length comparison
    foodspec_lines = len(inspect.getsource(run_foodspec_workflow).splitlines())
    manual_lines = len(inspect.getsource(manual_preprocess_and_features).splitlines()) + len(
        inspect.getsource(run_manual_workflow).splitlines()
    )
    print("\n=== Comparison ===")
    print(f"foodspec workflow lines (approx): {foodspec_lines}")
    print(f"Manual workflow lines (approx): {manual_lines}")
    print(f"foodspec accuracy (train-fit CV table): {food_result.cv_metrics['accuracy'].mean():.3f}")
    print(f"Manual accuracy (cross_val_predict): {manual_result['acc']:.3f}")
    print(f"foodspec confusion matrix:\n{food_result.confusion_matrix}")
    print(f"Manual confusion matrix:\n{manual_result['cm']}")

