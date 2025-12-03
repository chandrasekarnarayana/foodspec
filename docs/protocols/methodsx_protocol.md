# Protocols: MethodsX Mapping (Oil Authentication Example)

This chapter provides a MethodsX-style protocol for Raman oil authentication using FoodSpec. It covers materials, data acquisition, organization, analysis steps, and expected outcomes, and maps them to the `foodspec` CLI and API. Adapt the same template to FTIR/NIR or other domains.

## Objective
- Authenticate edible oils (e.g., differentiate olive, sunflower, canola) and detect adulteration using Raman spectra.
- Produce reproducible, report-ready outputs (metrics, confusion matrices, PCA plots) aligned with MethodsX/FAIR principles.

## Materials and instruments (generic)
- Raman spectrometer (e.g., 532/633/785 nm) with standard cuvette or vial holder.
- Calibration standards (e.g., silicon wafer for Raman shift check).
- Samples: known oils for training; blind or suspect oils for testing.
- Computer with Python 3.10+, FoodSpec installed (`pip install foodspec`).
- Optional: public/open dataset (e.g., Mendeley edible oils) converted to HDF5.

## Sample preparation (generic)
1. Homogenize oils; avoid bubbles/particulates.
2. Transfer to clean vial/cuvette; minimize fluorescence by choosing appropriate laser/wavelength/power.
3. Acquire replicate spectra per sample to capture variability.

## Data acquisition guidelines
- Spectral range: fingerprint (≈600–1800 cm⁻¹); optionally CH stretches (2800–3100 cm⁻¹).
- Resolution: sufficient to resolve unsaturation/ester bands (e.g., 4–8 cm⁻¹).
- Export: vendor TXT/CSV; one spectrum per file or a wide CSV. Ensure wavenumber axis is ascending.

## Data organization
- Folder or HDF5 library containing:
  - `x`: spectra (n_samples × n_wavenumbers), ascending cm⁻¹.
  - `wavenumbers`: axis.
  - `metadata`: columns `sample_id`, `oil_type`, optional `batch`, `instrument`.
  - `modality`: `"raman"` or `"ftir"`.
- Recommended: Convert CSV → HDF5 using `foodspec csv-to-library` (see [CSV → HDF5 pipeline](../csv_to_library.md)).

## Analysis pipeline (CLI)
1. Convert raw CSV to HDF5 (if needed):
   ```bash
   foodspec csv-to-library data/oils.csv libraries/oils.h5 \
     --format wide --wavenumber-column wavenumber --label-column oil_type --modality raman
   ```
2. Run oil authentication:
   ```bash
   foodspec oil-auth libraries/oils.h5 --label-column oil_type --output-dir runs/oil_auth
   ```
   Outputs: `metrics.json`, `confusion_matrix.png`, `report.md`, `run_metadata.json`.

## Statistical analysis steps
- After classification, test key ratios across oil types:
  - Run one-way ANOVA on ratios (e.g., 1655/1742) vs oil_type using `foodspec.stats.run_anova`.
  - Apply Tukey HSD (if statsmodels installed) to identify differing pairs.
  - Compute effect sizes (eta-squared/partial) where sums of squares are available.
- For regression/adulteration tasks, correlate predicted mixture fractions with ground truth; report RMSE/MAE and Pearson/Spearman correlations.
- Report alpha level, effect sizes, and confidence intervals where possible.

## Analysis pipeline (Python)
```python
from foodspec.data.loader import load_example_oils
from foodspec.apps.oils import run_oil_authentication_workflow

fs = load_example_oils()  # or load_library("libraries/oils.h5")
res = run_oil_authentication_workflow(fs, label_column="oil_type", classifier_name="rf", cv_splits=5)
print(res.cv_metrics)
```

## Steps and rationale
- **Preprocessing:** ALS baseline → Savitzky–Golay smoothing → L2 normalization → crop 600–1800 cm⁻¹.
- **Features:** Peaks around 1655, 1742, 1450 cm⁻¹; ratios 1655/1742, 1450/1655.
- **Models:** Random Forest (robust default); alternatives: SVM/PLS-DA.
- **Validation:** Stratified k-fold CV (5x); report accuracy, macro F1; confusion matrix.
- **Statistics:** ANOVA/Tukey on key ratios to corroborate model-driven discrimination; effect sizes to convey magnitude.

## Expected outcomes
- Clear separation of oil types in PCA; confusion matrix with high diagonal entries.
- Typical metrics: macro F1 in the high 0.8–0.9 range on clean datasets; lower if classes are subtle or data noisy.
- Artifacts generated: metrics.json, run_metadata.json, confusion_matrix.png, optional feature importances.
- Statistical tests should show significant differences in key ratios across oil types; post-hoc identifies specific pairs.

## Notes and tips
- Check baselines/fluorescence; adjust ALS parameters if over/under-correcting.
- Ensure wavenumbers are aligned and ascending; mismatches cause feature shifts.
- For adulteration levels (regression), consider mixture analysis (NNLS/PLS) instead of pure classification.

## Mapping to figures/tables
- **Scores/loadings:** Use PCA plots for exploratory figures.
- **Confusion matrix:** Main classification figure.
- **Metrics table:** Include accuracy, macro F1; per-class metrics in Supplementary.
- **Preprocessing/model config:** Include in methods text or Supplementary.

## Validation and reproducibility
- Capture software versions, seeds, CV splits, preprocessing choices (see [Reproducibility checklist](reproducibility_checklist.md)).
- Use `run_metadata.json` and saved configs for audits.

## Further reading
- [Reproducibility checklist](reproducibility_checklist.md)
- [Benchmarking framework](benchmarking_framework.md)
- [Oil authentication workflow](../workflows/oil_authentication.md)
