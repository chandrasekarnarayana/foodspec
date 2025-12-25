# Protocols: Reproducibility Checklist

Use this checklist to ensure FoodSpec analyses are transparent and repeatable. Include it in methods/protocol papers and QC documentation.

## Checklist (record/attach)
- **Data provenance:**
  - Dataset name/version; source/DOI.
  - File format (CSV → HDF5); wavenumber ordering (ascending cm⁻¹).
  - Metadata columns (sample_id, labels, conditions).
- **Preprocessing:**
  - Baseline method + parameters (ALS λ, p).
  - Smoothing (Savitzky–Golay window/poly), derivatives if used.
  - Normalization (L2/area/internal-peak), scatter correction (SNV/MSC), cropping ranges.
  - Modality corrections (ATR, atmospheric, cosmic ray removal).
- **Features:**
  - Expected peaks/ratios; band integration ranges; fingerprint similarity settings.
- **Models & validation:**
  - Model type + hyperparameters; seeds.
  - CV design (stratified folds, groups/batches); metrics reported.
  - Thresholds for QC/novelty if applicable.
- **Artifacts:**
  - metrics.json, run_metadata.json, report.md, plots.
  - Model registry entries (path, version, foodspec_version).
  - Plot/report flags used (e.g., pca_scores, confusion_matrix, feature_importances, spectra overlays; summary_json, markdown_report, run_metadata export).
- **Environment:**
  - Python/OS; package versions; CLI command/config used.
  - Hardware notes (GPU/CPU not required here, but note if used).
- **Statistics:**
  - Tests run (e.g., ANOVA, t-test, correlation), alpha level.
  - Effect sizes reported (Cohen’s d, eta-squared).
  - Design summary (group sizes, replication, randomization).
- **Citation & licensing:**
  - Software citation (CITATION.cff); dataset licenses/DOIs.

## Example snippet (fill in per study)
- Instrument: Raman, 785 nm, resolution 4 cm⁻¹; silicon calibration daily.
- Samples: oils (olive/sunflower), 10 spectra/class; randomized acquisition.
- Preprocessing: ALS (λ=1e5, p=0.01), Savitzky–Golay (win=9, poly=3), L2 norm, crop 600–1800 cm⁻¹.
- Features: peaks 1655/1742/1450, ratios 1655/1742, 1450/1655.
- Analysis: RF classifier (n_estimators=200), stratified 5-fold CV; stats: ANOVA on ratios (alpha=0.05), Tukey post-hoc; effect size: eta-squared.
- Metrics: accuracy 0.90 ± 0.02, macro F1 0.88 ± 0.03; ANOVA p < 0.01; effect size eta² = 0.45.
- Figures: confusion_matrix.png, pca_scores.png, boxplot ratios.
- Artifacts: metrics.json, run_metadata.json, report.md, configs logged; model registry entry: models/oil_rf_v1 (foodspec 0.2.0).

## Example (oil authentication, synthetic)
- Data: `libraries/oils.h5`, labels `oil_type`, ascending cm⁻¹.
- Preprocessing: ALS (λ=1e5, p=0.01), Savitzky–Golay (win=9, poly=3), L2 norm, crop 600–1800 cm⁻¹.
- Features: Peaks 1655/1742/1450 with ratios 1655/1742, 1450/1655.
- Model: Random Forest (n_estimators=200, random_state=0), Stratified 5-fold CV, metrics: accuracy, macro F1.
- Artifacts: `runs/oil_auth_demo/metrics.json`, `confusion_matrix.png`, `run_metadata.json`, `report.md`.
- Environment: Python 3.12, foodspec 0.2.0, OS Linux; CLI command recorded; seeds fixed.
- Citation: foodspec software (CITATION.cff).

## Notes
- Store checklists with run artifacts for audits.
- Prefer configs/YAML to capture parameters; avoid manual re-entry.

## Further reading

- [Benchmarking framework](benchmarking_framework.md)
- [Reporting guidelines](../troubleshooting/reporting_guidelines.md)
