# Standard Workflow Templates

This page lists concise templates you can adapt for common tasks. Each template references the relevant detailed workflow pages and points to troubleshooting/metrics/ML chapters.

## Authentication / Classification
- **Goal:** Identify class (e.g., oil type) or detect adulteration.
- **Template:**
  1. Load spectra (CSV/JCAMP/OPUS) with `read_spectra`.
  2. Preprocess: baseline → smoothing → normalization → crop.
  3. Features: peaks/ratios + optional PCA.
  4. Model: SVM/RF (or logreg baseline).
  5. Metrics: accuracy, F1_macro, confusion matrix; PR/ROC as needed.
  6. Reports: confusion matrix + per-class metrics; export run metadata/model.
- **See:** [Oil authentication](oil_authentication.md), [ML & metrics](../ml/models_and_best_practices.md).

## Adulteration (rare events)
- Same as authentication, but emphasize class imbalance:
  - Use class weights, PR curves, F1_macro; collect more positives.
  - Consider OC-SVM/IsolationForest for novelty.
- **See:** [Batch QC / novelty](batch_quality_control.md), [Troubleshooting](../troubleshooting/common_problems_and_solutions.md).

## Calibration / Regression
- **Goal:** Predict continuous quality/mixture values.
- **Template:**
  1. Preprocess consistently (baseline, norm, crop).
  2. Feature space: raw spectra, ratios, or PLS components.
  3. Model: PLS regression; consider MLP if non-linear bias remains.
  4. Metrics: RMSE, MAE, R², MAPE; plots: calibration + residuals.
  5. Robustness: bootstrap/permutation; check bias across range.
  6. Reports: predicted vs true, residual plots, parameter settings.
- **See:** [Calibration example](calibration_regression_example.md), [Metrics](../metrics/metrics_and_evaluation.md).

## Time/temperature trends (heating degradation)
- **Goal:** Track degradation markers vs time/temperature.
- **Template:** ratios vs time → trend models (linear/ANOVA) → slopes/p-values → plots (line + CI, box/violin by stage).
- **See:** [Heating quality monitoring](heating_quality_monitoring.md), [Stats](../stats/anova_and_manova.md).

## Mixtures
- **Goal:** Estimate component fractions.
- **Template:** NNLS with pure refs or MCR-ALS → metrics (RMSE/R²) → predicted vs true/residual plots.
- **See:** [Mixture analysis](mixture_analysis.md).

## Hyperspectral mapping
- **Goal:** Spatial localization.
- **Template:** per-pixel preprocessing → cube rebuild → ratios/PCs → clustering/classification → maps + pixel metrics.
- **See:** [Hyperspectral mapping](hyperspectral_mapping.md).

## Reporting essentials
- Record preprocessing parameters, model choices, metrics with uncertainty, plots, and configs; export run metadata/model artifacts.
- Consult [Reporting guidelines](../reporting_guidelines.md) and [Troubleshooting](../troubleshooting/common_problems_and_solutions.md) when issues arise.
