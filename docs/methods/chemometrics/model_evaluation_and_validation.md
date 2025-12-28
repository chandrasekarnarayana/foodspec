# ML & Chemometrics: Model Evaluation and Validation

Robust evaluation is essential for trustworthy food spectroscopy models. This page follows the WHAT/WHY/WHEN/WHERE template and adds concrete guidance for visualizing cross-validation (CV) results.

> For notation see the [Glossary](../../reference/glossary.md). Metrics: [Metrics & Evaluation](../../reference/metrics_reference.md).

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

---

## When Results Cannot Be Trusted

⚠️ **Red flags for validation design and model evaluation:**

1. **Data leakage in preprocessing (mean/std computed on entire dataset before train/test split)**
   - Information from test set influences training, inflating metrics
   - Leakage can be subtle; preprocessing should be inside CV loop
   - **Fix:** Use sklearn Pipeline to chain preprocessing + model; fit only on training folds; compute statistics on training data only

2. **Same data used for hyperparameter tuning and final evaluation**
   - Hyperparameters optimized on test set produce inflated performance estimates
   - Proper workflow: use training set for tuning, held-out test set for final evaluation
   - **Fix:** Use nested CV (outer folds for evaluation, inner folds for tuning) or separate tune/test sets

3. **Stratification not applied to small, imbalanced datasets**
   - Random train/test splits of imbalanced data can yield train set with even worse imbalance
   - Causes high fold-to-fold variability
   - **Fix:** Use stratified CV (StratifiedKFold); ensures all folds have similar class distribution

4. **Metrics reported without uncertainty (accuracy = 0.92, no confidence interval or SD across folds)**
   - Point estimates hide fold-to-fold variability; single fold may differ significantly
   - No uncertainty makes it impossible to assess significance of differences between models
   - **Fix:** Report mean ± SD across folds; compute bootstrap CI; show per-fold metrics in plot

5. **Batch structure ignored in CV (all samples from Device A in train, all from Device B in test)**
   - Temporal or instrument drift confounds model learning
   - Model may learn device artifacts, not generalizable patterns
   - **Fix:** Use GroupKFold to keep batches together in splits; validate across batch/device boundaries

6. **Perfect metrics on test set (accuracy 1.0, AUC 1.0) without investigation**
   - Too-perfect results suggest overfitting, data leakage, or class separation artifacts
   - Real food data rarely separates perfectly
   - **Fix:** Check for leakage; visualize test set; validate on completely independent, external data

7. **Class-specific metrics not reported (overall accuracy = 0.90, but minority class recall = 0.10)**
   - Aggregate metrics mask poor performance on minority class
   - Misleading for imbalanced tasks
   - **Fix:** Report per-class precision/recall/F1; use confusion matrix; consider weighted F1 or balanced accuracy

8. **Cross-validation with single fold or unrepeated splits**
   - 1-fold CV gives no sense of variability; non-repeated splits depend on random seed
   - Small datasets need more folds (5–10) for stable estimates
   - **Fix:** Use at least 5-fold CV; consider RepeatedStratifiedKFold for small datasets; report mean across repeats

9. **Temporal structure ignored (time-series data: train on future, test on past)**
   - Leakage through time leads to optimistic metrics
   - Temporal CV (train on earlier times, test on later) is more realistic
   - **Fix:** Use time-aware CV (TimeSeriesSplit) for sequential data; no forward-looking information in training
- [Workflows](../../workflows/authentication/oil_authentication.md)
