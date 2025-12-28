# Designing & Reporting Spectral Workflows

This chapter explains what a workflow means in FoodSpec and how to design, execute, and report it. A workflow connects the scientific question, instrument, data acquisition, preprocessing, feature extraction, statistics/ML, interpretation, and reporting.

## What is a workflow in FoodSpec?
```mermaid
flowchart LR
  subgraph Data
    A[Scientific question]
    B[Experimental design & instrument]
    C[Data acquisition & organization]
  end
  subgraph Preprocess
    D[Baseline / smoothing / norm / crop]
    E[Features: peaks, ratios, PCA/PLS]
  end
  subgraph Model_Stats
    F[Stats & ML (tests, classifiers, regressors, mixtures)]
  end
  subgraph Report
    G[Metrics + plots + report.md + run_metadata]
  end
  A --> B --> C --> D --> E --> F --> G
```
- It is a pipeline from question → instrument → data → preprocessing → features → analysis → interpretation → reporting.
- It aligns scientific goals with acquisition conditions, preprocessing choices, analysis methods, and clear outputs.
- Model and metric choices are covered in [ML & DL models](../methods/chemometrics/models_and_best_practices.md) and [Metrics & evaluation](../reference/metrics_reference.md).
- For issues at any stage (baseline, SNR, imbalance, overfitting), see [Common problems & solutions](../troubleshooting/common_problems_and_solutions.md).

## How to design a workflow (step-by-step)
1. **Clarify the question.** Examples: “Can we distinguish oil types?”, “How does heating affect quality?”, “Is this batch within spec?”
2. **Identify response variables and factors.** Class labels (oil_type), continuous metrics (peroxide value, mixture fraction), time/temperature, batches.
3. **Consider sample types/matrices.** Pure oils vs oils in chips; liquid vs powder; expected heterogeneity.
4. **Choose instrument and acquisition conditions.** Spectral range, resolution, SNR, laser power/integration time; ATR vs transmission; calibration routines.
5. **Plan preprocessing and features.** Baseline/smoothing/normalization; peaks/ratios, PCA/PC scores; see preprocessing chapters.
6. **Plan statistics/ML.** Are you comparing groups (ANOVA/t-test), predicting labels (classification), estimating fractions (regression/NNLS), detecting outliers (QC)?
7. **Plan visualization and reporting.** What plots/figures and metrics will demonstrate your answer?

### Questions to ask at each stage
- **Scientific goal:** What decision/action will this analysis support? What is “good enough” performance?
- **Instrument & acquisition:** What range/resolution is needed? What SNR? Are calibration and baselines stable? Any fluorescence, saturation, or drift?
- **Data structure:** How many samples per group? Are groups balanced? Any batch effects or confounders? Randomization?
- **Analysis:** Comparing groups, predicting labels, quantifying mixtures, detecting novelty? Expect linear/monotonic trends or complex patterns?
- **Reporting:** What should a reviewer reconstruct? Which methods, parameters, metrics, and figures must be included?

## Tools & methods in FoodSpec (questions → approaches)
| Question | Typical approach | FoodSpec modules/functions | Example workflow |
| --- | --- | --- | --- |
| Differentiate classes (e.g., oil types) | Classification (RF/SVM/PLS-DA), ANOVA on key ratios | `apps.oils`, `preprocess.*`, `features.*`, `stats.run_anova` | Oil authentication |
| Track degradation over time | Ratios vs time, correlation/linear regression, ANOVA across stages | `apps.heating`, `stats.compute_correlations`, `stats.run_anova` | Heating quality monitoring |
| Estimate mixture proportions | NNLS/PLS regression, correlation of predicted vs true | `chemometrics.mixture`, `stats.compute_correlations` | Mixture analysis |
| Screen batches for anomalies | One-class SVM/IsolationForest, t-tests on key metrics | `apps.qc`, `stats.run_ttest` | Batch QC |
| Assess associations | Pearson/Spearman, cross-correlation for sequences | `stats.compute_correlations`, `compute_cross_correlation` | Heating/time series |
| Visualize results | Confusion matrices, PCA, correlation heatmaps, calibration plots | `viz.confusion_matrix`, `viz.plot_pca_scores`, `viz.plot_correlation_heatmap`, `viz.plot_regression_calibration` | Across workflows |

When assumptions are doubtful or samples are small/skewed, use nonparametric tests (`run_mannwhitney_u`, `run_kruskal_wallis`, `run_wilcoxon_signed_rank`, `run_friedman_test`) and robustness checks (`bootstrap_metric`, `permutation_test_metric`) from `foodspec.stats`.

For instrument/file formats and ingestion routes, see the [Instrument & file formats guide](../user-guide/vendor_formats.md).

See also:
- Preprocessing: [Baseline](../methods/preprocessing/baseline_correction.md), [Normalization](../methods/preprocessing/normalization_smoothing.md), [Feature extraction](../methods/preprocessing/feature_extraction.md)
- ML: [Classification & regression](../methods/chemometrics/classification_regression.md), [Mixture models](../methods/chemometrics/mixture_models.md)
- Stats: [Hypothesis testing](../methods/statistics/hypothesis_testing_in_food_spectroscopy.md), [ANOVA/MANOVA](../methods/statistics/anova_and_manova.md)
- Protocols: [Reproducibility checklist](../protocols/reproducibility_checklist.md)
- API: [`foodspec.preprocess.baseline.ALSBaseline`](../api/preprocessing.md), [`foodspec.stats.run_anova`](../api/stats.md), [`foodspec.apps.oils.run_oil_authentication_workflow`](../08-api/workflows.md)

## Instrument specs, calibration, limitations → data
- **Range & resolution:** Determines which bands are captured/resolved. Choose ranges that cover key functional groups.
- **SNR & dynamic range:** Low SNR → more smoothing/replicates; saturation/bleed → adjust integration/laser power.
- **Laser power & integration time:** Affect intensity and fluorescence; document settings.
- **Optics (ATR vs transmission):** Influences depth of penetration and baseline shape.
- **Calibration:** Wavenumber calibration (e.g., silicon peak), intensity calibration if needed. Drift appears as shifts/baseline changes.
- **Artefacts:** Baseline drift, fluorescence, noise, cosmic rays. These motivate baseline correction, smoothing, normalization, and spike removal.

![Illustrative spectra: ideal vs noise vs baseline drift](../assets/spectra_artifacts.png)

## Reporting checklist (workflow level)
- **Instrument:** Model, spectral range/resolution, laser power/integration, calibration routine.
- **Samples:** Type/matrix, preparation, storage, number of replicates per group.
- **Acquisition:** Order/randomization, accumulations, environmental conditions.
- **Preprocessing:** Steps and parameters (baseline λ/p, smoothing window/poly, normalization choice, cropping).
- **Features:** Peak/ratio definitions, band ranges, PCs used.
- **Analysis:** Models, hyperparameters, validation strategy; statistical tests (alpha, effect sizes).
- **Metrics:** Accuracy/F1/ROC for classification; RMSE/MAE/R² for regression; test statistics/p-values/effect sizes for stats.
- **Figures:** Spectrum overlays, mean±CI spectra, PCA scores/loadings, confusion matrices, boxplots/violin plots, correlation plots, residuals.
- **Data/code:** Where to find raw/processed data, configs, run_metadata.json, reports.

## Plots & visualizations: what, why, and when {#plots-visualizations}
| Plot type | Main use | Example question | FoodSpec context |
| --- | --- | --- | --- |
| Raw/mean spectra | Inspect quality, baseline, noise | Are baselines drifting? | Preprocessing QC |
| PCA scores/loadings | Explore structure, drivers | Do oils cluster? Which bands matter? | Oil auth |
| Confusion matrix | Classification performance | Where does the classifier err? | Oil auth/QC |
| Box/violin plots | Group comparisons | Do ratios differ by type/stage? | Stats on features |
| Correlation scatter/heatmap | Association strength | Do ratios correlate with time/quality? | Heating/mixtures |
| Residual plots | Model diagnostics | Are residuals patterned? | Regression/mixtures |
| Time/temperature trends | Process monitoring | How do markers change with heating? | Heating workflow |

Expected details: axes labels (units), legends, group labels, sample sizes, confidence intervals/shading where relevant, consistent wavenumber units (cm⁻¹).

## Example mini-workflows
### Oil authentication (short narrative)
- Question: distinguish oil types/adulteration.
- Design: multiple oil classes, balanced reps; randomize acquisition; Raman fingerprint range.
- Acquisition: set laser/integration to avoid fluorescence/saturation; calibrate wavenumber.
- Preprocess: ALS baseline, Savitzky–Golay smoothing, L2 norm, crop.
- Features: peaks 1655/1742/1450; ratios.
- Analysis: RF classifier + stratified CV; ANOVA/Tukey on ratios; PCA for exploration.
- Plots: PCA scores/loadings, confusion matrix, boxplots of ratios.
- Reporting: metrics ± variability, test stats/effect sizes, preprocessing/model params.

### Heating quality monitoring (short narrative)
- Question: track degradation markers vs time/temperature.
- Design: time points/stages with replicates; control batches; randomize order if possible.
- Acquisition: consistent settings; monitor drift.
- Preprocess: baseline, smoothing, normalization, crop.
- Features: unsaturation ratio (1655/1742) vs time; optional other bands.
- Analysis: correlation/linear regression of ratio vs time; ANOVA if discretized stages; trend models.
- Plots: ratio vs time with fit, boxplots by stage, optional residuals.
- Reporting: slope/p-value, R², CIs; preprocessing/model details.

## Summary
- A workflow is a coherent pipeline from question to report; design drives instrument choice, preprocessing, analysis, and visualization.
- Ask the right questions at each stage; use FoodSpec tools (preprocessing, features, ML, stats) accordingly.
- Report instruments, preprocessing, features, analysis methods, metrics, and figures transparently.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for workflow validity:**

1. **Instrument or preprocessing not documented ("we used standard Raman; no special preprocessing")**
   - Results depend critically on instrument settings and preprocessing choices
   - Undisclosed choices prevent reproduction or cross-instrument validation
   - **Fix:** Document laser wavelength, power, integration time, baseline correction (parameters), smoothing, normalization, peak alignment

2. **No reference sample or control in workflow (no validation against known composition)**
   - Unknown whether conclusions apply to intended samples or are artifacts
   - Control validates method, detector, and preprocessing
   - **Fix:** Include positive control (known authentic/adulterant), negative control (pure reference), and blank (empty cuvette)

3. **Workflow tested only on same instrument/day/operator**
   - Results don't generalize to other labs or future analyses
   - Batch effects (drift, calibration) confound biological effects
   - **Fix:** Validate on data from different instruments, days, or operators; use batch-aware CV; test generalization

4. **Preprocessing parameters tuned on same data used for model evaluation**
   - Overfitting in preprocessing (hyperparameter tuning) inflates metrics
   - Parameters optimized on test data don't generalize
   - **Fix:** Freeze preprocessing parameters before data splitting; include preprocessing in CV loop

5. **Results meeting workflow goals without investigating outliers or failures**
   - Some samples may fail preprocessing (e.g., fluorescing oils, broken cuvette)
   - Ignoring failures overestimates success rate
   - **Fix:** Document failure modes; report success rate; investigate failed samples

6. **No comparison to baseline or previous method (reporting new workflow accuracy with no context)**
   - Accuracy/precision depend on problem difficulty; comparison shows improvement
   - New workflow may be worse than old if not benchmarked
   - **Fix:** Compare metrics to baseline (previous method, random guess, published results); report relative improvement

7. **Temporal structure ignored (time-series workflow: training on future data, testing on past)**
   - Leakage through time produces overoptimistic metrics
   - For trending workflows (heating quality), time-aware CV needed
   - **Fix:** Use temporal CV (train on past, test on future); no forward-looking leakage

8. **Interpretation relies only on model metrics without visual/chemical validation**
   - Metrics alone don't confirm biological/chemical validity
   - Workflow producing high accuracy on noise still worthless
   - **Fix:** Visualize outputs (spectra, residuals, predictions); validate top features chemically; involve domain experts

## See also
- [Workflows](authentication/oil_authentication.md)
- [Statistics overview](../methods/statistics/overview.md)
- [Preprocessing guides](../methods/preprocessing/baseline_correction.md)
- [Protocols & reproducibility](../protocols/reproducibility_checklist.md)
