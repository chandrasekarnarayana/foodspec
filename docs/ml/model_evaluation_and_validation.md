# ML & Chemometrics: Model Evaluation and Validation

Robust evaluation is essential for trustworthy food spectroscopy models. This page follows the WHAT/WHY/WHEN/WHERE template and adds concrete guidance for visualizing cross-validation (CV) results.

> For notation see the [Glossary](../glossary.md). Metrics: [Metrics & Evaluation](../metrics/metrics_and_evaluation.md).

## What?
Defines validation schemes (train/test, stratified CV, group-aware CV, permutation tests), the metrics to report, and how to visualize per-fold outcomes (confusion matrices, residuals, calibration).

## Why?
Spectral datasets are often small, imbalanced, or batch-structured. Validation guards against overfitting/leakage, provides uncertainty via fold variability, and underpins protocol-grade reporting.

## When?
**Use:** stratified k-fold for classification; group-aware CV when batches/instruments matter; permutation tests when checking above-chance performance.  
**Limitations:** tiny n inflates variance; imbalance makes accuracy unreliable; always scale/normalize within folds to avoid leakage.

## Where? (pipeline)
Upstream: fixed preprocessing/feature steps.  
Validation: CV/permutation.  
Downstream: metrics + plots + stats on key ratios.  
```mermaid
flowchart LR
  A[Preprocess + features] --> B[CV / permutation]
  B --> C[Metrics + per-fold plots]
  C --> D[Reporting + stats tables]
```

## Validation designs
- **Stratified k-fold (classification):** preserve class proportions.  
- **Group-aware CV:** avoid leakage across batches/instruments.  
- **Train/test split:** simple, less stable on small n.  
- **Permutation tests:** label-shuffle to test above-chance performance.  
- **Pitfalls:** normalize within folds; do not tune on test; document seeds/splits.

## Metrics (by task)
- Classification: F1_macro/balanced accuracy + confusion matrix; ROC/PR for imbalance.  
- Regression/calibration: RMSE/MAE/R²/Adjusted R² + predicted vs true + residuals; calibration with CI bands; Bland–Altman for agreement.  
- Embeddings: silhouette, between/within F-like stats with permutation p_perm (see metrics chapter).

## Visualizing CV folds (guidance replacing TODO)
Pattern: collect per-fold predictions and metrics, then plot distributions:
```python
from foodspec.chemometrics.validation import cross_validate_pipeline
from foodspec.viz import plot_confusion_matrix, plot_regression_calibration, plot_residuals

cv = cross_validate_pipeline(pipeline, X_feat, y_labels, cv_splits=5, scoring="f1_macro")
# Per-fold metrics
print(cv["metrics_per_fold"])  # e.g., list of F1s
# Example per-fold confusion matrix (if returned/recomputed)
plot_confusion_matrix(cv["confusion_matrices"][0], labels=class_labels)
```
- For regression folds: loop over folds, plot residuals or predicted vs true per fold, or aggregate predicted/true across folds and plot once.  
- For a quick visual summary of fold metrics: make a boxplot/violin of the per-fold metric list.

## Examples
### Classification (stratified CV)
```python
cv = cross_validate_pipeline(clf, X_feat, y_labels, cv_splits=5, scoring="f1_macro")
f1s = cv["metrics_per_fold"]
# visualize distribution of f1s with a simple boxplot (matplotlib/seaborn)
```
### Regression
```python
cv = cross_validate_pipeline(pls_reg, X_feat, y_cont, cv_splits=5, scoring="neg_root_mean_squared_error")
# After CV, refit on full data if appropriate; visualize calibration/residuals on a held-out set or via CV predictions.
```

## Sanity checks and pitfalls
- Very high scores on tiny n → suspect overfitting/leakage.  
- Imbalance → use macro metrics; inspect per-class supports.  
- Re-run with different seeds/folds to test stability; report mean ± std/CI across folds.  
- Keep preprocessing identical across folds; document seeds, splits, hyperparameters.

## Typical plots (with metrics)
- Confusion matrix (per fold or aggregate) + F1/accuracy/supports.  
- ROC/PR for rare-event tasks.  
- Predicted vs true + residuals for regression; calibration with CI (`plot_calibration_with_ci`).  
- Fold-metric distribution plot (box/violin of per-fold F1 or RMSE).

## Summary
- Choose validation design aligned with data structure (stratified, group-aware).  
- Pair metrics with uncertainty (fold variability, bootstrap CIs).  
- Avoid leakage; report seeds/splits/preprocessing.  
- Visualize per-fold behavior to reveal instability or class-specific failures.

## Further reading
- [Classification & regression](classification_regression.md)  
- [Metrics & evaluation](../metrics/metrics_and_evaluation.md)  
- [Reproducibility checklist](../protocols/reproducibility_checklist.md)  
- [Workflows](../workflows/oil_authentication.md)
