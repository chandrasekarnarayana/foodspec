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
- **See:** [Calibration example](calibration_regression_example.md), [Metrics](../metrics/metrics_and_evaluation/).

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
- Consult [Reporting guidelines](../troubleshooting/reporting_guidelines.md) and [Troubleshooting](../troubleshooting/common_problems_and_solutions.md) when issues arise.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for template-based workflows:**

1. **Using template with different domain/instrument without revalidation (oil template applied to dairy without testing)**
   - Templates are domain-specific; spectral signatures, sample prep, and matrix effects differ
   - Model trained on oils won't work on milk or meat
   - **Fix:** Validate template on target domain before use; test on 10+ samples to confirm applicability

2. **Template preprocessing parameters not adjusted for new matrix (using oil normalization on dairy proteins)**
   - Preprocessing optimal for one food type may be poor for another
   - Different absorbance ranges, solubility, fluorescence require different settings
   - **Fix:** Test preprocessing on new matrix; adjust parameters (baseline lambda, smoothing); freeze before analysis

3. **Template model directly deployed without cross-validation on new data (assuming old model works)**
   - Model trained on past data; generalization to new batches/instruments unverified
   - Drift, seasonal changes, or instrumental variation can invalidate model
   - **Fix:** Cross-validate model on representative new samples; retrain if performance drops >10%

4. **Features/ratios from template used blindly without domain interpretation**
   - Template features may be arbitrary (optimized for one dataset, not chemically meaningful)
   - Different food type may have different separating features
   - **Fix:** Validate template features are chemically plausible for new domain; check loadings/importance

5. **Metrics thresholds from template applied without local calibration (using template accuracy cutoff 0.85 for new domain)**
   - Template thresholds calibrated on template data; new domain may need adjustment
   - Class distributions and difficulty differ across domains
   - **Fix:** Recalibrate decision thresholds on new data; validate at operational point (sensitivity/specificity target)

6. **Template applied to imbalanced data without rebalancing**
   - Templates often assume balanced classes; imbalanced deployment inflates majority class accuracy
   - Minority class performance may be poor
   - **Fix:** Stratify CV; use class weights; retrain if classes severely imbalanced

7. **No documentation of why template was chosen or when it's appropriate**
   - Templates are gray boxes; unclear which template fits which problem
   - Can lead to inappropriate template selection
   - **Fix:** Document template scope (domain, food type, spectroscopy method); include decision flowchart

8. **Template results trusted without sensitivity analysis (no testing on edge cases or outliers)**
   - Templates may fail on unusual samples (off-spec, oxidized, contaminated)
   - Real samples have variability beyond template training distribution
   - **Fix:** Test template on challenging samples (old oils, mixed types, degraded); document failure modes
