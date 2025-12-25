# Chemometrics & models

This guide ties together the main chemometric models used in FoodSpec and how they connect to preprocessing, metrics, and statistics.

> For notation see the [Glossary](../09-reference/glossary.md). For practical bands/ratios, see [Feature extraction](../../preprocessing/feature_extraction/#how-to-choose-bands-and-ratios-decision-mini-guide). For metrics, see [Metrics & Evaluation](../../metrics/metrics_and_evaluation/).

## What?
Defines PCA/PLS projections, classifiers, and mixture models, with inputs (preprocessed spectra, labels/targets, or pure spectra) and outputs (scores/loadings, predictions, fractions, metrics).

## Why?
Spectra are high-dimensional and correlated; chemometrics reduces dimensionality, builds predictive models, and estimates mixtures while respecting spectroscopy constraints (non-negativity, correlated bands).

## When?
**Use when:** you need dimensionality reduction (PCA), calibration (PLS), supervised discrimination (LogReg/SVM/RF/kNN/PLS-DA), or mixture estimation (NNLS/MCR-ALS).  
**Limitations:** small n → prefer simpler models + CIs; strong nonlinearity → consider RBF SVM/MLP; mixture models assume non-negative combinations of appropriate references.

## Where? (pipeline)
Upstream: consistent preprocessing (baseline → smoothing → normalization → optional derivatives/features).  
Downstream: metrics (F1, AUC, RMSE), stats tests (ANOVA/Games–Howell on ratios/PCs), interpretability (loadings/importances), reporting.
```mermaid
flowchart LR
  A[Preprocessing + features] --> B[Chemometric model]
  B --> C[Metrics + stats tests]
  C --> D[Interpretation + reporting]
```

## PCA (scores & loadings)
- Purpose: reduce dimensionality, visualize clustering, identify spectral regions driving variance.  
- Outputs: scores (sample coordinates) and loadings (wavenumber weights); explained variance via scree.  
- Interpretation: clusters in scores → similarity/differences; high loadings near bands → chemically relevant drivers. Pair with silhouette/between-within metrics from the metrics chapter.

## PLS and PLS-DA
- PLS regression links spectra \(X\) to y (continuous); PLS-DA couples PLS scores with a classifier.  
- Good for correlated predictors and modest sample size; validate with CV and report RMSE/R² (and calibration plots with CI).

## Classifiers (selection)
- Logistic regression: linear, fast baseline.  
- SVM (linear/RBF): margin maximization; RBF handles nonlinear boundaries.  
- Random Forest: nonlinear, feature importances.  
- Gradient boosting (sklearn) and boosted trees (optional XGBoost/LightGBM via `pip install foodspec[ml]`): strong tabular performance and good with nonlinear interactions; tune learning rate/trees.  
- kNN: instance-based, sensitive to scaling/imbalance.  
Choose by data size/linearity/interpretability; always report per-class metrics and pair with ROC/PR where appropriate.

## Mixture models (NNLS, MCR-ALS)
- NNLS: for one mixture spectrum \(\mathbf{x}\) and pure spectra matrix \(\mathbf{S}\), solve \(\min_{\mathbf{c}\ge0}\|\mathbf{x}-\mathbf{S}\mathbf{c}\|_2^2\); \(\mathbf{c}\) are non-negative fractions.
- MCR-ALS: for multiple mixtures \(\mathbf{X}\), factorize \(\mathbf{X}\approx \mathbf{C}\mathbf{S}^\top\) with non-negativity; returns concentrations \(\mathbf{C}\) and estimated pure spectra \(\mathbf{S}\).
- Diagnostics: reconstruction RMSE/R² and residual plots (see [Mixture models](../ml/mixture_models.md)).

## Validation helpers
- Cross-validation (stratified for classification) to estimate generalization.  
- Permutation tests (optionally) for label-randomization checks.  
- Metrics: F1/ROC–AUC/confusion for classification; RMSE/MAE/R² for regression/mixture; silhouette/between-within for embeddings.

## Practical guidance
- Keep preprocessing identical between train/test; scale features for distance-based models.  
- Stratify or balance classes; report supports and macro metrics.  
- Pair visuals (PCA/score plots, reconstruction overlays) with quantitative metrics + CIs.  
- Prefer simpler models when performance is similar to ease interpretation and reproducibility.  
- Inspect residuals for regression/mixture; large structure in residuals implies missing bands or model mismatch.  
- For heteroscedastic groups, use Games–Howell for post-hoc comparisons on ratios/PCs (see [ANOVA & MANOVA](../stats/anova_and_manova.md)).

## See also
- [Metrics & Evaluation](../../metrics/metrics_and_evaluation/)  
- [Model evaluation & validation](../ml/model_evaluation_and_validation.md)  
- [Mixture models](../ml/mixture_models.md)  
- [Model interpretability](../ml/model_interpretability.md)  
- [Workflow design & reporting](../workflows/workflow_design_and_reporting.md)
