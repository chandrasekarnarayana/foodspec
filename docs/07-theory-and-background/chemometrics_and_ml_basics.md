# Theory – Chemometrics & ML Basics

This page summarizes core concepts underpinning FoodSpec analyses. For worked examples, see Tutorials (02) and Cookbook (03).

## Core methods
- **PCA**: unsupervised dimensionality reduction; reveals structure/clusters and supports clustering metrics (silhouette/ARI).
- **PLS/PLS-DA**: regression/classification with latent variables (not always needed for simple ratio sets but common in spectroscopy).
- **Classification**: logistic regression (often with L1 for minimal panels), random forests for nonlinear importance; balanced accuracy/confusion matrices for evaluation.

## Why cross-validation matters
- Prevents optimistic bias; estimates generalization performance.
- Batch-aware or group-aware splits avoid leakage across instruments/batches.
- Nested CV supports feature selection/hyperparameter tuning without reusing test folds.

## Scaling/normalization
- Standardization and ratiometric features stabilize intensity variations; see preprocessing recipes and RQ theory for why specific ratios are used.

See also: [cookbook_validation.md](../03-cookbook/cookbook_validation.md) and [oil_discrimination_basic.md](../02-tutorials/oil_discrimination_basic.md) for applied examples.

How FoodSpec uses these:
- PCA/MDS visualizations and clustering metrics in RQ outputs.
- Classification (LR/RF) for discrimination and minimal panels.
- Cross-validation strategies (batch-aware/nested) for honest performance estimates.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for chemometrics and ML theory application:**

1. **High-dimensional data (p >> n) without regularization or dimensionality reduction**
   - Overfitting guaranteed; model memorizes noise
   - Unstable coefficients and poor generalization
   - **Fix:** Use PCA before classification; use regularized models (Ridge, Lasso); enforce p < n or use cross-validation

2. **Eigenvalues/variance explained not examined (using PCs without checking explained variance)**
   - PC5 explaining <1% variance is likely noise; including it overfits
   - Cumulative variance plateaus; using components beyond plateau adds noise
   - **Fix:** Plot scree plot; use cumulative variance rule (e.g., 95%) to choose n_components

3. **Collinearity among features not detected (using all spectral bands as predictors without checking VIF)**
   - Correlated features inflate coefficients and reduce stability
   - Model unstable to small data perturbations
   - **Fix:** Compute VIF; remove/group collinear features; use PCA or regularization

4. **Linear model assumptions not checked before using linear methods (using PLS on non-linear data)**
   - Linear methods assume linear relationships; non-linear data requires non-linear models
   - Predictions biased; confidence intervals unreliable
   - **Fix:** Visualize feature-target relationships; use non-linear models if relationships non-linear

5. **Class imbalance ignored in classification (95% class A, 5% class B, standard classifier applied)**
   - Classifier biased toward majority; minority class ignored
   - Standard metrics (accuracy) misleading; minority class F1 can be near zero
   - **Fix:** Use stratified CV; class weights; report per-class metrics

6. **Distance metrics not matched to data type (Euclidean distance on compositional data)**
   - Euclidean distance assumes continuous, not-compositional data
   - Can produce misleading clustering
   - **Fix:** Use compositional distances (Aitchison, log-ratio) or appropriate metrics for data type

7. **Scaling applied inconsistently (training scaled, test not scaled, or vice versa)**
   - Feature magnitudes mismatch between train and test
   - Models produce wrong predictions on unscaled test data
   - **Fix:** Always apply same scaling to train and test; include scaling in pipeline

8. **Latent factors interpreted causally ("PC1 is oxidation" without validation)**
   - PCs are mathematical combinations; may not correspond to chemistry
   - Interpretation requires independent validation
   - **Fix:** Cross-check PC loadings with chemistry; validate interpreted factors with targeted measurements
