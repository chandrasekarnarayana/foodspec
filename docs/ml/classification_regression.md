# ML & Chemometrics: Classification and Regression

Supervised learning connects spectral features to labels (oil_type, species) or continuous targets (mixture fraction, heating time proxies). This chapter explains model choices, when to use them, and practical concerns in food spectroscopy.

## Why this matters in food spectroscopy
- Authentication/adulteration: classification with balanced or imbalanced classes.
- QC/novelty: one-class or binary detection with emphasis on sensitivity/specificity.
- Calibration: regression for continuous properties (mixture fraction, moisture, quality indices).

## 1. Model families
- **Linear models:** Logistic Regression, Linear SVM, PLS-DA. Good for linearly separable data, interpretable loadings/weights, smaller datasets.
- **Non-linear kernels:** RBF SVM captures curved boundaries; tune C and gamma carefully. k-NN is simple but sensitive to scale and noise.
- **Ensembles:** Random Forest, Gradient Boosting (if enabled) handle non-linearities and interactions; often robust to noisy features.
- **Regression:** PLS regression for mixtures or property prediction; Support Vector Regression (SVR) for non-linear trends.

## 2. When to use / not to use
- **Use linear/PLS-DA when:** You want interpretability; dataset is modest; features are ratios/peaks; classes are approximately linear.
- **Use RBF SVM/RF when:** Complex decision boundaries; higher-dimensional feature sets; you can afford tuning/validation.
- **Avoid high-capacity models when:** Very small datasets risk overfitting; noisy preprocessing; extreme class imbalance without care.

## 3. Cross-validation and metrics
- **CV design:** Stratified k-fold for classification; grouped CV if batches exist. Small datasets: fewer folds to keep enough training data.
- **Metrics:** Accuracy/F1_macro for imbalance; confusion matrix for per-class errors; ROC/PR for rare positives. Regression: RMSE, MAE, R², calibration/residual plots.
- **Pitfalls:** Data leakage (normalize per fold), tuning on test sets, over-interpreting small performance gaps.
- See [Metrics & evaluation](../metrics/metrics_and_evaluation.md) and Stats ([hypothesis testing](../stats/hypothesis_testing_in_food_spectroscopy.md), [effect sizes](../stats/t_tests_effect_sizes_and_power.md)) when comparing models.

## 4. Example (high level)
```python
from foodspec.chemometrics.models import make_classifier
from foodspec.chemometrics.validation import cross_validate_pipeline

clf = make_classifier("svm_rbf", C=10.0, gamma=0.1)
cv_res = cross_validate_pipeline(clf, X_feat, y_labels, cv_splits=5, scoring="f1_macro")
```
CLI analogue: `foodspec oil-auth --classifier-name svm_rbf --cv-splits 5`.

### Regression / calibration example
```python
import numpy as np
import pandas as pd
from foodspec.chemometrics.models import make_pls_regression
from foodspec.chemometrics.validation import compute_regression_metrics

# X_feat: shape (n_samples, n_features); y_cont: continuous target (e.g., mixture fraction)
pls = make_pls_regression(n_components=3)
pls.fit(X_feat, y_cont)
y_pred = pls.predict(X_feat).ravel()
metrics = compute_regression_metrics(y_cont, y_pred)
print(metrics)  # reports RMSE, MAE, R^2
```

![Regression calibration plot: predicted vs true values](../assets/regression_calibration.png)

*Figure: Example regression calibration plot showing predicted vs true values for a PLS regression model on synthetic data. Points close to the diagonal line indicate good calibration; systematic deviation signals bias or underfitting/overfitting. See the [Calibration / regression workflow](../workflows/calibration_regression_example.md) for an end-to-end example with robustness checks.*

## 5. Practical notes for food spectra
- **Class imbalance:** Common in adulteration tasks; use macro metrics, possibly class weights.
- **Feature scaling:** Ensure consistent preprocessing; derivatives/ratios can change scale.
- **Interpretation:** Link model importance/loadings back to wavenumbers (e.g., unsaturation bands) for scientific insight.

## 6. Visuals to include
- Confusion matrix (classification), ROC/PR curves for imbalance (use `plot_confusion_matrix`, `plot_roc_curve`).
- Predicted vs true plots (regression) and residuals; calibration curve (bias/slope) (use `plot_regression_calibration`, `plot_residuals`).
- For regression/calibration, pair curves with uncertainty: use `plot_calibration_with_ci` for confidence bands and, when comparing methods, consider Bland–Altman plots (see [Metrics & evaluation](../metrics/metrics_and_evaluation.md)).
- Feature importance or model coefficients/loadings for interpretability.

## Reproducible figure generation
- Classification: run a simple PCA+SVM on example oils, then call `plot_confusion_matrix` (optionally ROC/PR if scores are available).
- Regression: use the PLS example above and generate calibration/residual plots via `foodspec.viz`.
- For agreement assessment, add Bland–Altman when comparing model vs reference lab values.

## Summary
- Choose linear/PLS-DA for interpretability and smaller datasets; non-linear models for complex boundaries.
- Use stratified CV and appropriate metrics; avoid leakage and overfitting.
- Tie model outputs back to chemical meaning (bands/ratios) for reporting.

## Further reading
- [PCA and dimensionality reduction](pca_and_dimensionality_reduction.md)
- [Mixture models & fingerprinting](mixture_models.md)
- [Model evaluation & validation](model_evaluation_and_validation.md)
- [Workflows](../workflows/oil_authentication.md)
- [Metrics & evaluation](../metrics/metrics_and_evaluation.md)
- [Hypothesis testing](../stats/hypothesis_testing_in_food_spectroscopy.md)
