# Preprocessing Guide

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Users processing raw Raman/FTIR spectra who need to remove background noise and normalize intensities.  
**What problem does this solve?** Understanding which preprocessing steps to apply and when, avoiding common pitfalls.  
**When to use this?** Before running classification/regression workflows; when spectra have baseline drift, noise, or scaling issues.  
**Why it matters?** Proper preprocessing is critical—poor preprocessing leads to unreliable models and false conclusions.  
**Time to complete:** 20-30 minutes to read; actual preprocessing takes seconds to minutes depending on dataset size.  
**Prerequisites:** Raw spectral data (CSV or HDF5); basic understanding of spectroscopy artifacts; FoodSpec installed

---

## What is Preprocessing?

Preprocessing makes real spectra comparable by removing background, reducing noise, and normalizing scale. Below are the major steps implemented in FoodSpec, the problems they solve, and when to use them.

## Baseline correction
- **ALS (Asymmetric Least Squares)**: iteratively fits a smooth baseline; subtracts fluorescence (Raman) or sloping background (FTIR).  
  - Math: penalized spline with asymmetry parameter p to favor fitting the lower envelope.  
  - Use when broad background dominates peaks; avoid overfitting (tune λ, p).
- **Rubberband**: connects convex hull of the spectrum and subtracts it.  
  - Good for concave baselines; less effective with strong convex fluorescence.
- **Polynomial**: low-degree polynomial fit to the whole spectrum.  
  - Use when baseline is globally smooth and simple; avoid high degree to prevent peak distortion.

## Smoothing
- **Savitzky–Golay**: local polynomial regression; preserves peak shape.  
  - Parameters: odd window length; polyorder < window_length.  
  - Use for moderate noise; avoid very short windows that under-smooth or long windows that flatten peaks.
- **Moving average**: simple mean over a window.  
  - Use for gentle denoising; may broaden peaks.

## Normalization and scatter correction
- **Vector/Area/Max normalization**: scale spectra to unit norm or unit area to remove intensity scaling differences.  
  - Use for consistent scaling; avoid if absolute intensity carries meaning.
- **SNV (Standard Normal Variate)**: subtract mean and divide by std per spectrum.  
  - Removes additive/multiplicative effects; use on diffuse reflectance or when scatter varies; avoid if spectra have near-zero variance.
- **MSC (Multiplicative Scatter Correction)**: regress each spectrum onto a reference (mean spectrum) and correct slope/intercept.  
  - Use for scatter differences; requires representative reference; avoid if reference is poor or spectra have zeros/NaNs.
- **Internal-peak normalization**: scale so a known internal band has mean 1 within a window.  
  - Use when a stable internal standard exists (e.g., known peak unaffected by composition).

## Cropping
- Restrict to informative regions (e.g., 600–1800 cm⁻¹ fingerprint).  
- Reduces noise and irrelevant regions; essential when instrument provides wide ranges.

## FTIR-specific helpers
- **AtmosphericCorrector**: subtracts water/CO₂ components using template bases; useful for MIR air-path measurements.  
- **SimpleATRCorrector**: compensates ATR depth effects with a heuristic scaling; use cautiously, mainly for comparative studies.

## Raman-specific helper
- **CosmicRayRemover**: detects sharp spikes and interpolates; apply to Raman spectra with cosmic-ray artifacts.

## Example default pipelines
- **Raman**: ALS (λ ~ 1e5, p ~ 0.01) → Savitzky–Golay (window 9, poly 3) → Vector norm → Crop 600–1800 cm⁻¹ → (optional) CosmicRayRemover before baseline.  
- **FTIR (ATR)**: Rubberband or ALS → Savitzky–Golay (gentle) → MSC or SNV → Optional Atmospheric/ATR corrections → Crop to target region (e.g., 900–1800 cm⁻¹).  

## When it may cause problems
- Over-aggressive baseline can distort broad peaks.
- Too-strong smoothing can flatten narrow peaks.
- Normalization can hide absolute concentration effects.
- Scatter corrections assume linear relationships; not ideal if chemical changes alter band shapes dramatically.

## When Results Cannot Be Trusted

Preprocessing results are unreliable when:
1. **Over-aggressive baseline correction** – Peaks become distorted or inverted; check by comparing before/after plots
2. **Excessive smoothing** – Narrow peaks disappear or broaden significantly; verify peak widths remain consistent
3. **Inappropriate normalization** – Absolute intensity information is lost when it's scientifically meaningful
4. **Poor scatter correction reference** – MSC with non-representative reference spectrum introduces artifacts

**How to detect:**
- Visual inspection: plot raw vs preprocessed spectra for representative samples
- Peak integrity check: verify known peaks maintain expected shape and relative intensities
- Batch consistency: ensure preprocessing doesn't introduce systematic batch differences

**What to do:**
- Tune preprocessing parameters (λ, p for ALS; window size for smoothing)
- Use validation metrics on downstream tasks to evaluate preprocessing effectiveness
- Document preprocessing choices and parameters for reproducibility

---

## Cross-References

For more details see:
- FTIR/Raman-specific methods: [ftir_raman_preprocessing.md](ftir_raman_preprocessing.md)
- Chemometrics workflows: [chemometrics_guide.md](chemometrics_guide.md)
- Validation strategies: [cookbook_validation.md](cookbook_validation.md)
