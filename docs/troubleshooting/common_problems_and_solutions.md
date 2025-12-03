# Troubleshooting: Common Problems & Solutions

Questions this page answers:
- What can go wrong in spectroscopy workflows and how do I detect it?
- How do I diagnose issues with FoodSpec tools (plots, metrics, utilities)?
- How do I fix or mitigate problems, and when should I re-acquire data?

This chapter groups common problems by stage: instrument/acquisition → dataset/metadata → preprocessing/chemometrics → ML/DL → statistics → visualization → reporting → workflow design → operational errors.

## A. Instrument & Acquisition Problems
**Baseline drift / fluorescence**
- Why: sample fluorescence, laser instability, optics heating.
- Symptoms: sloping/curved baseline; high low-frequency power.
- Diagnose: overlay raw spectra; run baseline check after ALS/rubberband; SNR via `estimate_snr`.
- Fix: apply baseline correction (ALS/rubberband); reduce laser power/integration time; instrument recalibration.
- Re-acquire: if baseline consumes dynamic range or varies wildly run-to-run.

**Saturation / clipping**
- Why: detector overload, too high laser power.
- Symptoms: flat-topped peaks, abrupt ceiling.
- Diagnose: inspect raw intensities; histogram of intensities.
- Fix: lower laser power, shorten integration time; re-acquire if clipping is present.

**Wavenumber misalignment**
- Why: calibration drift, temperature, instrument change.
- Symptoms: peak shifts vs references.
- Diagnose: compare known standards; cross-correlation of spectra.
- Fix: recalibrate instrument; apply alignment/cropping consistently; re-acquire if shift unstable.

**Low SNR**
- Why: weak scattering/absorption, poor focus, dirty optics.
- Symptoms: noisy spectra, unstable ratios.
- Diagnose: `estimate_snr`; high-frequency noise; low reproducibility across replicates.
- Fix: longer integration, more accumulations, better sample prep/optics cleaning; smoothing; re-acquire if SNR too low.

## B. Dataset & Metadata Problems
**Missing or inconsistent metadata**
- Why: incomplete logs, manual entry errors.
- Symptoms: unknown labels, mismatched sample IDs.
- Diagnose: `check_missing_metadata`; cross-check unique counts; joins fail.
- Fix: repair metadata files; enforce required columns; re-export if gaps persist.

**Class imbalance**
- Why: rare adulteration/spoilage cases.
- Symptoms: high accuracy, poor minority recall.
- Diagnose: `summarize_class_balance`; confusion matrix asymmetry; PR curves.
- Fix: resampling/weights, use F1/PR metrics; collect more minority samples.

**Mislabeled samples**
- Why: data entry or sample mix-up.
- Symptoms: persistent outliers, impossible confusion errors.
- Diagnose: PCA score outliers; `detect_outliers`; high leverage points.
- Fix: audit sample IDs; remove/relable after verification; re-acquire if uncertain.

## C. Preprocessing & Chemometric Problems {#c-preprocessing--chemometric-problems}
**Over-smoothing / under-smoothing**
- Symptoms: peak loss or excessive noise.
- Diagnose: compare raw vs smoothed overlays; SNR changes.
- Fix: adjust Savitzky–Golay window/order; avoid smoothing if not needed.

**Baseline not removed / over-corrected**
- Symptoms: residual slope or negative artifacts.
- Diagnose: inspect corrected spectra; mean spectrum drift.
- Fix: tune ALS lambda/p; try rubberband/polynomial; ensure crop before ratios.

**Scatter/normalization issues**
- Symptoms: intensity scaling differences remain.
- Diagnose: norms variance across samples; check after SNV/MSC/vector norms.
- Fix: use SNV/MSC; ensure consistent application within pipelines (no leakage).

**Peak picking / ratios unstable**
- Symptoms: large variance in peak height/area; missing peaks.
- Diagnose: visualize peak windows; check wavenumber alignment; inspect window tolerance.
- Fix: adjust expected peaks/tolerance; ensure ascending wavenumbers; consider smoothing/cropping first.

## D. Machine Learning Problems
**Overfitting**
- Symptoms: high train accuracy, low test/CV accuracy.
- Diagnose: CV metrics vs train; learning curves.
- Fix: simplify model, regularize, more data, better preprocessing; ensure stratified CV; use `compute_classification_metrics`.

**Data leakage**
- Symptoms: unrealistically high CV scores.
- Diagnose: verify preprocessing inside Pipeline; splits done after pipeline definition; no label leakage.
- Fix: wrap preprocessing+model in a single pipeline; redo splits; re-evaluate.

**Imbalanced performance**
- Symptoms: minority class misclassified.
- Diagnose: confusion matrix by class; PR curves; class balance summary.
- Fix: class weights, resampling, threshold tuning; report F1_macro, balanced accuracy.

## E. Deep Learning Problems
**Unstable training / divergence**
- Symptoms: loss oscillations, NaNs.
- Diagnose: monitor loss/metrics per epoch; check learning rate/batch size.
- Fix: lower learning rate, use normalization, add early stopping/dropout; ensure sufficient data.

**Overfitting with small data**
- Symptoms: train ≫ val performance.
- Diagnose: validation curves; high variance metrics.
- Fix: regularize, data augmentation (if appropriate), prefer classical models.

## F. Statistical Problems
**Violating test assumptions**
- Symptoms: non-normal residuals, heteroscedasticity.
- Diagnose: residual plots, normality tests, Levene’s test.
- Fix: transform data, use nonparametric tests (`run_kruskal_wallis`, `run_mannwhitney_u`); report effect sizes.

**Multiple comparisons without correction**
- Symptoms: many marginal p-values.
- Diagnose: count of tests; inconsistent significance.
- Fix: use Tukey/FDR; emphasize effect sizes; consolidate hypotheses.

## G. Visualization Problems
**Misleading scales / unlabeled axes**
- Symptoms: hard-to-read plots; ambiguous units.
- Diagnose: review plots; check legends/units.
- Fix: label wavenumber (cm⁻¹), intensity (a.u.), class labels, sample counts; use consistent ranges.

**Overplotting / clutter**
- Symptoms: unreadable overlays with many samples.
- Diagnose: high-density overlays.
- Fix: show mean ± CI, subset samples, use transparency.

## H. Reporting & Reproducibility Problems
**Missing pipeline/config trace**
- Symptoms: cannot reproduce metrics or plots later.
- Diagnose: absent configs, missing run_metadata.json.
- Fix: use `export_run_metadata`; record preprocessing, models, metrics, versions.

**Ambiguous metrics**
- Symptoms: headline accuracy without class counts or CI.
- Diagnose: incomplete reporting.
- Fix: include per-class metrics, supports, CIs/bootstraps; link to `metrics_and_evaluation`.

## I. Workflow Design Problems
**Unclear question → wrong pipeline**
- Symptoms: metrics irrelevant to decision (e.g., accuracy on rare event).
- Diagnose: revisit scientific goal; map task to metrics/models.
- Fix: consult [workflow design](../workflows/workflow_design_and_reporting.md); pick appropriate metrics/models.

**Insufficient replicates / imbalance**
- Symptoms: unstable metrics across splits.
- Diagnose: high variance CV; `summarize_class_balance`.
- Fix: collect more data; use robust CV; consider effect sizes and uncertainty reporting.

## J. Operational / User Errors
**Wrong file format / path**
- Symptoms: loader failures.
- Diagnose: check `detect_format`, file extensions; consult instrument file formats guide.
- Fix: convert to supported formats (CSV, JCAMP, SPC/OPUS with extras).

**Mismatched wavenumber ordering**
- Symptoms: shape errors, misaligned peaks.
- Diagnose: ensure ascending wavenumbers; validate with `validate_spectrum_set`.
- Fix: sort wavenumbers; re-export if needed.

---
### FoodSpec utilities for diagnosis
- `estimate_snr(spectrum)`: rough SNR estimate.
- `summarize_class_balance(labels)`: counts per class.
- `detect_outliers(X, method="pca_distance")`: simple outlier flagging.
- `check_missing_metadata(df, required_cols)`: ensure metadata completeness.

### When to re-acquire data
- Severe saturation/clipping; unstable baselines consuming dynamic range.
- Wavenumber calibration drift not correctable in software.
- Extremely low SNR that preprocessing cannot salvage.
- Persistent metadata mislabeling that cannot be resolved.
