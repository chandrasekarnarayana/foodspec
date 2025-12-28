# ML & Chemometrics: Classification and Regression

Supervised learning connects spectral features to labels (oil_type, species) or continuous targets (mixture fraction, heating proxies). This page follows the WHAT/WHY/WHEN/WHERE template for clarity.

> For notation see the [Glossary](../../reference/glossary.md). For bands/ratios, see [Feature extraction](../preprocessing/feature_extraction.md). Metrics: [Metrics & Evaluation](../../reference/metrics_reference.md).

## What?
Defines model families for classification (LogReg, SVM, RF, kNN, PLS-DA) and regression/calibration (PLS, SVR), with inputs (preprocessed features/ratios/PCs) and outputs (predictions, probabilities/scores, calibration curves, metrics).

## Why?
Authentication/adulteration, QC/novelty, and calibration require models that handle correlated bands and small-to-medium n. Supervised models turn spectral variation into decisions or calibrated estimates, and must be paired with suitable metrics/plots to avoid overclaiming.

## When?
**Use:**  
- Linear/PLS-DA: modest n, interpretable boundaries, ratios/PCs roughly linear.  
- RBF SVM/RF: nonlinear boundaries, richer feature sets.  
- PLS/linear/SVR: continuous targets, mixtures, property prediction.  
**Limitations:**  
- Small n: risk of overfitting—use CV + CIs.  
- Imbalance: accuracy can mislead—use macro F1/PR.  
- Always standardize consistently; leakage if scaling across folds.

## Where? (pipeline)
Upstream: baseline/smoothing/normalization → optional derivatives → features/ratios/PCs.  
Model: classifier/regressor.  
Downstream: metrics (F1/AUC/RMSE/R² + CIs), plots (confusion, ROC/PR, calibration/residual), stats tests on key ratios (ANOVA/Games–Howell), reporting.
```mermaid
flowchart LR
  A[Preprocess + features] --> B[Classifier / Regressor]
  B --> C[Metrics + plots + stats]
  C --> D[Interpretation & reporting]
```

## Model families (at a glance)
- **Linear / PLS-DA**: interpretable coefficients/loadings; good for smaller, near-linear problems.  
- **SVM (linear/RBF)**: max-margin; RBF handles curved boundaries (tune C, gamma).  
- **Random Forest / Ensembles**: nonlinear, feature importances; robust to noisy predictors.  
- **Boosted trees (GradientBoosting / XGBoost / LightGBM)**: strong tabular performance; handle interactions and imbalance; require tuning (learning rate, trees).  
- **kNN**: simple baseline; sensitive to scaling/imbalance.  
- **PLS regression / SVR**: calibration and property prediction; pair with calibration plots + residuals.

## Metrics and plots (pair visuals with numbers)
- Classification: F1_macro, balanced accuracy, confusion matrix; ROC/PR for imbalance (see `plot_confusion_matrix`, `plot_roc_curve`).  
- Regression/calibration: RMSE/MAE/R²/Adjusted R², predicted vs true, residuals; `plot_calibration_with_ci` for confidence bands, `plot_bland_altman` for agreement.  
- Embeddings: silhouette, between/within F-like stats with permutation p_perm alongside PCA/t-SNE visuals.

## Examples
### Classification (SVM)
```python
from foodspec.chemometrics.models import make_classifier
from foodspec.chemometrics.validation import cross_validate_pipeline
from foodspec.viz import plot_confusion_matrix

clf = make_classifier("svm_rbf", C=10.0, gamma=0.1)
cv_res = cross_validate_pipeline(clf, X_feat, y_labels, cv_splits=5, scoring="f1_macro")
plot_confusion_matrix(cv_res["confusion_matrix"], labels=class_labels)
```

### Regression / calibration (PLS)
```python
from foodspec.chemometrics.models import make_pls_regression
from foodspec.metrics import compute_regression_metrics
from foodspec.viz import plot_regression_calibration, plot_residuals

pls = make_pls_regression(n_components=3)
pls.fit(X_feat, y_cont)
y_pred = pls.predict(X_feat).ravel()
metrics = compute_regression_metrics(y_cont, y_pred)
plot_regression_calibration(y_cont, y_pred)  # add CI with plot_calibration_with_ci if desired
plot_residuals(y_cont, y_pred)
```
![Regression calibration plot: predicted vs true values](../../assets/regression_calibration.png)

## Practical notes for food spectra
- Imbalance: use macro metrics, class weights, and PR curves for rare positives.  
- Scaling: apply the same scaling/derivative steps per fold; avoid leakage.  
- Interpretation: map coefficients/loadings/importances back to bands (unsaturation, carbonyl) and report ANOVA/Games–Howell on key ratios when relevant.  
- Validate: stratified CV; report supports and CIs; inspect residuals for bias or structure.

## Typical plots (with metrics)
- Confusion matrix + F1/accuracy/supports.  
- ROC/PR + AUC (especially for rare-event adulteration).  
- Predicted vs true + residuals + RMSE/R² (calibration).  
- Calibration curve with CI; Bland–Altman for agreement.  
- Feature importances/loadings to link decisions to wavenumbers.

## Reproducible figure generation
- Classification: PCA + SVM on example oils, then `plot_confusion_matrix` (ROC/PR if scores available).  
- Regression: PLS on example mixtures; generate calibration/residual plots (`plot_regression_calibration`, `plot_calibration_with_ci`, `plot_residuals`).  
- Agreement: `plot_bland_altman` when comparing model vs lab measurements.

## Summary
- Match model complexity to data size/linearity; prefer interpretable models when performance is similar.  
- Combine visuals with metrics + uncertainty; avoid leakage; handle imbalance explicitly.  
- Tie outputs back to chemistry (bands/ratios) and support claims with stats tests on key features.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for classification and regression model reliability:**

1. **Perfect or near-perfect accuracy (>98%) without independent test set**
   - Too-good results suggest overfitting, data leakage, or class separation artifacts
   - Accuracy on training data inflates; test set is ground truth
   - **Fix:** Always use held-out test set or cross-validation; validate on truly independent data

2. **Data leakage (using preprocessing statistics from entire dataset, or same sample in train and test)**
   - Information from test set bleeds into training, producing inflated metrics
   - Leakage can be subtle (e.g., baseline correction fit on all data, then train/test split)
   - **Fix:** Include preprocessing inside CV loop; use Pipeline to prevent leakage; manually inspect train/test splits

3. **Severe class imbalance (95% class A, 5% class B) with no special handling**
   - Imbalanced data can produce misleading accuracy (classifier predicts majority class, achieves 95% accuracy)
   - F1, precision, recall, or AUC better than accuracy for imbalanced tasks
   - **Fix:** Stratify CV splits; use weighted loss or resampling; report confusion matrix and per-class metrics

4. **High feature dimensionality, small sample size (1000 bands, 50 samples) without dimensionality reduction**
   - p >> n (features > samples) causes overfitting and unstable models
   - Model has enough parameters to memorize training data
   - **Fix:** Use PCA or feature selection before modeling; keep p < n or use regularization (L1/L2)

5. **Regression with huge residuals in certain regions unaddressed**
   - Heteroscedasticity (variance increases with predicted value) violates assumptions
   - May indicate nonlinear relationship or missing important feature
   - **Fix:** Visualize predicted vs residuals; check for heteroscedasticity; consider log-transform or nonlinear model

6. **Feature importance from same model used for training (overfitted features list)**
   - Feature importance from overfit model is unreliable; top features may be noise in that model
   - Importance doesn't generalize to independent data
   - **Fix:** Use permutation importance on test set; or report with uncertainty (e.g., from cross-validation)

7. **No attention to batch effects (training on one analyzer, deploying on another)**
   - Analyzer drift, calibration, or instrument variation can dominate learned patterns
   - Model may not transfer across instruments without retraining
   - **Fix:** Use batch-aware CV; test on data from different batches/instruments; include batch as a feature

8. **Unstable cross-validation metrics (fold 1 F1 = 0.95, fold 5 F1 = 0.50)**
   - High fold-to-fold variability suggests model is sensitive to data splits or presence of outliers
   - Reported average metric masks instability
   - **Fix:** Visualize per-fold metrics; check for outliers; increase sample size; use stratified CV
- [Model evaluation & validation](model_evaluation_and_validation.md)  
- [Metrics & evaluation](../../reference/metrics_reference.md)  
- [Hypothesis testing](../statistics/hypothesis_testing_in_food_spectroscopy.md)  
- [Workflow design & reporting](../../workflows/workflow_design_and_reporting.md)
