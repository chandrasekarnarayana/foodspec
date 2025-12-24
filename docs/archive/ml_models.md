# Machine learning models in foodspec

> **Status:** Legacy/archived. This page is superseded by `../ml/models_and_best_practices.md` and the mkdocstrings API pages. Retained here for historical reference.

Questions this page answers
- Which models are available and when to use them?
- How do I invoke them via foodspec (classifier_name/run_pca)?
- What should I report?

## Linear models
- **Logistic regression**: linear boundary; good baseline, interpretable.  
  - Foodspec: `classifier_name="logreg"` in workflows.  
  - Example:
    ```python
    from foodspec.chemometrics.models import make_classifier
    clf = make_classifier("logreg")
    clf.fit(X_train, y_train)
    ```
  - Report: accuracy/F1, confusion matrix; note regularization if tuned.
- **Linear SVM**: margin maximization for linear problems.  
  - Foodspec: `classifier_name="svm_linear"`.
- **PLS-DA**: PLS projection + classifier for correlated spectra.  
  - Foodspec: `make_pls_da(n_components=10)`.

## Non-linear models
- **RBF SVM**: nonlinear separation.  
  - Foodspec: `classifier_name="svm_rbf"`; key params: C, gamma.
- **k-NN**: instance-based; sensitive to scaling/imbalance.  
  - Foodspec: `classifier_name="knn"`; key param: k.
- **Random Forest (RF)**: ensemble of trees; robust, feature importance.  
  - Foodspec: `classifier_name="rf"`; key params: n_estimators, max_depth.
- **Gradient boosting** (if configured): stronger ensemble; risk overfitting on small sets.
- Example (via workflow):
  ```python
  clf = make_classifier("rf", n_estimators=200, random_state=0)
  clf.fit(X_train, y_train)
  ```

## Unsupervised methods
- **PCA**: dimensionality reduction/visualization; use for QC/exploration.  
  - Foodspec: `run_pca(X, n_components=2)`.  
  - Report explained variance and score plots.
- **Clustering**: not core in foodspec; apply to PCA/features if needed (k-means, etc.).

## Mixture models
- **NNLS / MCR-ALS**: estimate component fractions in mixtures.  
  - Foodspec: `nnls_mixture`, `mcr_als` in `chemometrics.mixture`.  
  - Report RMSE/R², residuals.

## Deep learning (optional)
- **Conv1DSpectrumClassifier**: 1D CNN prototype; optional extra (`foodspec[deep]`).  
  - Use only with sufficient data; risk of overfitting on small sets.  
  - Report train/val split, regularization, seeds.

## Reporting
- State model type and key hyperparameters.
- Report metrics: accuracy, macro/weighted F1, ROC-AUC (if probabilities), confusion matrix; for regression/mixture, R²/RMSE.
- Use stratified CV for classification; report mean ± std.
- Mention preprocessing and features (baseline/smoothing/normalization, peaks/ratios).

See also
- [Metrics & evaluation](../metrics/metrics_and_evaluation.md)
- [Oil authentication tutorial](../workflows/oil_authentication.md)

- [API index](../api/index.md)
# ML models (legacy)

> **Status: Legacy / Archived**  
> This page has been superseded by [Models & best practices](../ml/models_and_best_practices.md) and [Classification & regression](../ml/classification_regression.md). It is kept for historical reference.
