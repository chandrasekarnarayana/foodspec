# Chemometrics & models

This guide provides intuition for the main methods used in FoodSpec to extract information from spectra.

> For notation and symbols, see the [Glossary](glossary.md). For practical bands/ratios, see the mini-guide in [Feature extraction](preprocessing/feature_extraction.md#how-to-choose-bands-and-ratios-decision-mini-guide).

## PCA (Principal Component Analysis)
- Purpose: reduce dimensionality, visualize clustering, and identify spectral regions driving variance.  
- Outputs: **scores** (sample coordinates in PC space) and **loadings** (weights per wavenumber).  
- Explained variance: fraction of total variance captured by PCs; use scree plots to pick PCs.  
- Interpretation: clusters in score plots suggest similarity; loadings show which bands contribute most.

## PLS / PLS-DA
- PLS (regression) links spectra (X) to a continuous response (y), maximizing covariance.  
- PLS-DA combines PLS projection with a classifier (e.g., logistic regression).  
- Good for correlated predictors and modest sample sizes; watch for overfitting—use cross-validation.

## Classifiers (intuitive)
- **Logistic regression**: linear boundary; fast baseline.  
- **SVM (linear/RBF)**: maximizes margin; RBF handles nonlinear separations.  
- **Random forest**: ensemble of trees; captures nonlinearities and can rank feature importance.  
- **kNN**: instance-based; sensitive to scaling and class imbalance.  
Choose based on data size, linearity, and need for interpretability.

## Mixture models (NNLS, MCR-ALS)
- **NNLS**: for one mixture spectrum \(\mathbf{x}\) and pure spectra matrix \(\mathbf{S}\), solve \(\mathbf{x} \approx \mathbf{S}\mathbf{c}\) with \(c_i \ge 0\). Coefficients \(\mathbf{c}\) are estimated fractions.  
- **MCR-ALS**: for multiple mixtures matrix \(\mathbf{X}\), factorize \(\mathbf{X} \approx \mathbf{C}\mathbf{S}^\top\) iteratively with non-negativity. Retrieves concentrations \(\mathbf{C}\) and pure-like spectra \(\mathbf{S}\).

## Validation helpers
- Cross-validation (stratified for classification) to estimate generalization.
- Metrics: accuracy, F1, confusion matrices for classification; R²/RMSE for regression/mixture.
- Permutation tests (if used) to assess significance by label shuffling.

## Practical guidelines
- Always keep preprocessing identical between train/test.
- Stratify when classes are imbalanced; report per-class metrics.
- Inspect residuals for regression/mixture tasks to detect bias.
- Prefer simpler models if performance is similar—easier to explain and reproduce.
