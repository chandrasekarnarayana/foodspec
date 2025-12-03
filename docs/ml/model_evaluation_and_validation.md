# ML & Chemometrics: Model Evaluation and Validation

Robust evaluation is essential for trustworthy food spectroscopy models. This chapter summarizes validation design, metrics, and sanity checks for classification and regression tasks.

## 1. Why validation matters
- Prevents overfitting and false confidence, especially on small datasets.
- Ensures reported performance reflects generalization to new batches/instruments.
- Supports reproducibility and protocol-grade reporting.

## 2. Cross-validation designs
- **Stratified k-fold (classification):** Preserve class proportions; common default.
- **Group-aware CV:** If batches/instruments exist, group by batch to avoid leakage.
- **Train/test split:** Simple, but less stable on small datasets.
- **Permutation tests:** Assess whether performance is above chance.
- **Pitfalls:** Normalize within folds; avoid tuning on the test set; document seeds.

## 3. Classification metrics
- **Accuracy:** Overall correct fraction; can mislead with imbalance.
- **Precision/Recall/F1 (macro):** Class-averaged; robust to imbalance.
- **Confusion matrix:** Per-class error patterns; critical for adulteration detection.
- **ROC-AUC/PR-AUC:** Optional for probabilistic models; beware of small-sample noise.

## 4. Regression metrics
- **RMSE/MAE:** Absolute error scales; MAE is robust to outliers.
- **R²:** Variance explained; watch for negative values on poor fits.
- **Residual analysis:** Plot residuals vs predicted/true; look for bias or heteroscedasticity.

## 5. Example (high level)
```python
from foodspec.chemometrics.validation import cross_validate_pipeline

cv = cross_validate_pipeline(pipeline, X_feat, y_labels, cv_splits=5, scoring="f1_macro")
print(cv["mean"], cv["std"])
```
> TODO: Add convenience plotting for confusion matrices/residuals linked to CV folds.

## 6. Sanity checks and pitfalls
- Very high scores with tiny datasets → likely overfitting or leakage.
- Imbalance → prefer macro/weighted metrics; inspect per-class performance.
- Re-run with different seeds/folds to test stability.
- Document preprocessing, model hyperparameters, seeds, and data splits.

## 7. Visuals to include
- Confusion matrix with per-class labels.
- Residual plots; predicted vs true scatter for regression.
- Metric distributions across folds.

## Summary
- Validation design (CV, grouping) and metric choice are as important as the model itself.
- Use macro metrics and confusion matrices for classification; RMSE/MAE/R² and residuals for regression.
- Avoid leakage; report seeds, splits, and preprocessing steps for reproducibility.

## Further reading
- [Classification & regression](classification_regression.md)
- [Metrics interpretation](../metrics_interpretation.md)
- [Reproducibility checklist](../protocols/reproducibility_checklist.md)
- [Workflows](../workflows/oil_authentication.md)
