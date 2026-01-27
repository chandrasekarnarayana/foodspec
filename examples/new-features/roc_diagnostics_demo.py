#!/usr/bin/env python
"""Demo: ROC/AUC Diagnostics for Model Evaluation

This example demonstrates the new ROC/AUC diagnostics module, showing:
- Binary classification ROC analysis with bootstrap confidence intervals
- Multiclass classification with per-class, micro, and macro AUC metrics
- Threshold optimization (Youden's J statistic)
- Integration with FitPredictResult from the modeling API

Run with: python roc_diagnostics_demo.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from foodspec.modeling import (
    compute_roc_diagnostics,
    compute_roc_for_result,
    fit_predict,
)


def demo_binary_roc():
    """Demonstrate binary ROC analysis."""
    print("\n" + "=" * 70)
    print("DEMO 1: Binary Classification ROC Analysis")
    print("=" * 70)

    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Compute ROC diagnostics with bootstrap CI
    print("\nComputing ROC diagnostics with 1000 bootstrap replicates...")
    result = compute_roc_diagnostics(y_test, y_proba, n_bootstrap=1000, random_seed=42)

    # Access metrics
    metrics = result.per_class[1]
    print("\n📊 Binary ROC Results:")
    print(f"   AUC: {metrics.auc:.4f}")
    print(f"   95% CI: [{metrics.ci_lower:.4f}, {metrics.ci_upper:.4f}]")
    print(f"   Positives: {metrics.n_positives}, Negatives: {metrics.n_negatives}")

    # Optimal threshold
    if "youden" in result.optimal_thresholds:
        youden = result.optimal_thresholds["youden"]
        print("\n🎯 Youden's J Optimal Threshold:")
        print(f"   Threshold: {youden.threshold:.4f}")
        print(f"   Sensitivity: {youden.sensitivity:.4f}")
        print(f"   Specificity: {youden.specificity:.4f}")
        print(f"   J-statistic: {youden.j_statistic:.4f}")


def demo_multiclass_roc():
    """Demonstrate multiclass ROC analysis."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multiclass Classification ROC Analysis (OvR)")
    print("=" * 70)

    # Generate synthetic multiclass data
    X, y = make_classification(
        n_samples=300,
        n_features=25,
        n_informative=15,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train multiclass classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)

    # Compute multiclass ROC
    print("\nComputing multiclass ROC diagnostics...")
    result = compute_roc_diagnostics(y_test, y_proba, n_bootstrap=500, random_seed=42)

    print("\n📊 Multiclass ROC Results:")
    print("   Per-class AUCs:")
    for class_label, metrics in result.per_class.items():
        print(f"      Class {class_label}: {metrics.auc:.4f} (CI: [{metrics.ci_lower:.4f}, {metrics.ci_upper:.4f}])")

    print(f"   Macro-average AUC: {result.macro_auc:.4f}")
    print(f"   Micro-average AUC: {result.micro.auc:.4f}")

    # Explain the difference
    print("\n💡 Metric Interpretation:")
    print("   • Per-class: Binary ROC for each class vs. rest")
    print("   • Macro: Simple average across classes (equal weight)")
    print("   • Micro: Aggregated TP/FP across classes (sample-weighted)")


def demo_with_fitpredict():
    """Demonstrate ROC diagnostics integrated with fit_predict results."""
    print("\n" + "=" * 70)
    print("DEMO 3: ROC Diagnostics from fit_predict Results")
    print("=" * 70)

    # Generate data
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )

    # Train model with fit_predict
    print("\nTraining model with fit_predict (nested CV)...")
    result = fit_predict(
        X,
        y,
        model_name="logreg",
        scheme="nested",
        outer_splits=3,
        seed=42,
    )

    print(f"   Model trained on {len(result.y_true)} samples")
    print(f"   Classes: {result.classes}")

    # Compute ROC diagnostics from result
    print("\nComputing ROC diagnostics from fit_predict result...")
    roc_diag = compute_roc_for_result(result, n_bootstrap=500, random_seed=42)

    metrics = roc_diag["roc_result"].per_class[1]
    print("\n📊 ROC Diagnostics:")
    print(f"   AUC: {metrics.auc:.4f}")
    print(f"   95% CI: [{metrics.ci_lower:.4f}, {metrics.ci_upper:.4f}]")
    print(f"   Metadata: {roc_diag['metadata']}")


def demo_edge_cases():
    """Demonstrate handling of edge cases."""
    print("\n" + "=" * 70)
    print("DEMO 4: Edge Cases and Special Scenarios")
    print("=" * 70)

    # Case 1: Perfect separation
    print("\n1️⃣ Perfect Separation (AUC = 1.0):")
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = compute_roc_diagnostics(y_true, y_proba, n_bootstrap=100, random_seed=42)
    metrics = result.per_class[1]
    print(f"   AUC: {metrics.auc:.4f}")
    print(f"   CI: [{metrics.ci_lower:.4f}, {metrics.ci_upper:.4f}] (collapsed at boundary)")

    # Case 2: Random classifier
    print("\n2️⃣ Random Classifier (AUC ≈ 0.5):")
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 100)
    y_proba = np.random.uniform(0, 1, 100)
    result = compute_roc_diagnostics(y_true, y_proba, n_bootstrap=100, random_seed=42)
    metrics = result.per_class[1]
    print(f"   AUC: {metrics.auc:.4f}")
    print(f"   CI: [{metrics.ci_lower:.4f}, {metrics.ci_upper:.4f}]")

    # Case 3: Class imbalance
    print("\n3️⃣ Imbalanced Dataset (90% class 0, 10% class 1):")
    y_true = np.concatenate([np.zeros(90), np.ones(10)])
    y_proba = np.concatenate([
        np.random.uniform(0, 0.4, 90),  # Mostly low probs
        np.random.uniform(0.6, 1.0, 10),  # Mostly high probs
    ])
    result = compute_roc_diagnostics(y_true, y_proba, n_bootstrap=100, random_seed=42)
    metrics = result.per_class[1]
    print(f"   AUC: {metrics.auc:.4f}")
    print(f"   Positives: {metrics.n_positives}, Negatives: {metrics.n_negatives}")
    print("   Note: AUC is robust to class imbalance")


def demo_reproducibility():
    """Demonstrate reproducibility with fixed seeds."""
    print("\n" + "=" * 70)
    print("DEMO 5: Reproducibility with Fixed Random Seeds")
    print("=" * 70)

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    # Run twice with same seed
    print("\nRunning ROC computation twice with seed=42...")
    result1 = compute_roc_diagnostics(y_true, y_proba, n_bootstrap=100, random_seed=42)
    result2 = compute_roc_diagnostics(y_true, y_proba, n_bootstrap=100, random_seed=42)

    metrics1 = result1.per_class[1]
    metrics2 = result2.per_class[1]

    print("\n✓ Results are Identical:")
    print(f"   Run 1: AUC={metrics1.auc:.6f}, CI=[{metrics1.ci_lower:.6f}, {metrics1.ci_upper:.6f}]")
    print(f"   Run 2: AUC={metrics2.auc:.6f}, CI=[{metrics2.ci_lower:.6f}, {metrics2.ci_upper:.6f}]")
    print(f"   Exact match: {metrics1.auc == metrics2.auc and metrics1.ci_lower == metrics2.ci_lower}")

    # Run with different seed
    print("\nRunning with different seed (seed=123)...")
    result3 = compute_roc_diagnostics(y_true, y_proba, n_bootstrap=100, random_seed=123)
    metrics3 = result3.per_class[1]

    print("\n⚠️ Different seed produces different CI:")
    print(f"   Seed 42: CI=[{metrics1.ci_lower:.6f}, {metrics1.ci_upper:.6f}]")
    print(f"   Seed 123: CI=[{metrics3.ci_lower:.6f}, {metrics3.ci_upper:.6f}]")


if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print("ROC/AUC DIAGNOSTICS MODULE DEMONSTRATION")
    print("🎯" * 35)

    demo_binary_roc()
    demo_multiclass_roc()
    demo_with_fitpredict()
    demo_edge_cases()
    demo_reproducibility()

    print("\n" + "=" * 70)
    print("✅ All demonstrations completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Read the full documentation: docs/api/diagnostics.md")
    print("  2. Try with your own data and models")
    print("  3. Use compute_roc_for_result() to add diagnostics to your workflows")
