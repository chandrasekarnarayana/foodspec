# Reporting guidelines

**Purpose:** Document your analysis results reproducibly and support peer review.  
**Audience:** Researchers publishing FoodSpec workflows or sharing results.  
**Time to read:** 8–12 minutes.  
**Prerequisites:** Completed a FoodSpec workflow run; familiar with classification/regression metrics.

---

## What to present as core results
- **Main figures**: confusion matrix (classification), PCA scores, key ratio/time trends, predicted vs true plots (mixture/regression).  
- **Main tables**: overall accuracy/F1 (classification) or R²/RMSE (regression/mixture); include fold-averaged metrics.  
- **Preprocessing summary**: list methods/parameters (baseline, smoothing, normalization, cropping).  
- **Model and validation**: classifier/regressor type, CV design (folds, stratification), seeds.

## What belongs in supplementary material
- Full per-class precision/recall/F1 tables, additional confusion matrices.
- Hyperparameters, alternative models tried, sensitivity analyses.
- Additional spectra/ratios, extended residual plots, run_metadata.json/config files.

## Describing methods for reproducibility
- State data origin, modality, instrument, sample prep conditions.  
- Describe preprocessing steps with parameters; note wavenumber range.  
- Specify features used (peaks/ratios/bands) and model choices.  
- Document validation setup: CV splits, stratification, metrics reported, any held-out test sets.  
- Reference CLI/Python commands or configs used (e.g., `foodspec oil-auth` with flags, or script snippets).

## Follow-up/supporting tests
- Independent dataset or instrument for external validation.
- Orthogonal analyses (e.g., GC–MS, peroxide/anisidine values, sensory tests) to corroborate spectroscopy results.
- Robustness checks: new batches, different preprocessing, or small perturbations to confirm stability.

Align with FAIR: keep data + metadata together, cite public datasets/DOIs, share configs and run artifacts when possible.

---

## Next Steps

- [Troubleshooting & FAQs](troubleshooting_faq.md) — Quick fixes for common issues during analysis.
- [Common Problems & Solutions](common_problems_and_solutions.md) — Deep dive into diagnosis and remediation strategies.
- [Metrics Reference](../reference/metrics_reference.md) — Understand what each metric means for your report.
