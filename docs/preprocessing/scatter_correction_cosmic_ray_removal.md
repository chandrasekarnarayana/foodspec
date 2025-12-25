# Preprocessing: Scatter Correction and Cosmic Ray Removal

Scatter and spike artifacts can mask true chemical signals. This chapter explains scatter-aware corrections (ATR/atmospheric) and spike removal for Raman.

## 1. Scatter in FTIR/Raman
- **ATR-FTIR:** Variable contact and refractive-index mismatch produce sloping baselines and intensity changes.
- **Atmospheric effects:** Water/CO₂ bands superimpose on spectra.
- **Raman:** Spike-like cosmic rays from high-energy particles.

## 2. Corrections in FoodSpec (how it works)
### Atmospheric correction (FTIR)
- **Concept:** Fit/subtract water/CO₂ basis functions; scaled templates are removed from spectra.
- **Use when:** Working in open air or with noticeable water/CO₂ bands.
- **Pitfalls:** Over-subtraction can distort nearby peaks; validate visually.

### Simple ATR correction
- **Concept:** Heuristic scaling for effective path-length changes with wavelength and incidence angle.
- **Use when:** ATR contact is inconsistent; mild correction is sufficient.
- **Pitfalls:** Approximate; not a replacement for rigorous optical modeling.

### Scatter-aware normalization (SNV/MSC)
- See [Normalization & smoothing](../normalization_smoothing/); SNV/MSC mitigate path-length/contact effects via linear rescaling to a reference.

### Cosmic ray removal (Raman)
- **Concept:** Detect spikes far above local median/derivative thresholds; replace by local interpolation.
- **Use when:** Narrow, isolated spikes appear in Raman spectra.
- **Pitfalls:** Avoid mistaking narrow real peaks for spikes; tune thresholds conservatively.

## 3. When to use / not to use
- Use atmospheric/ATR correction for FTIR when environmental or contact effects are visible.
- Use cosmic-ray removal for spike artifacts; skip if spectra are already spike-free.
- Combine with baseline correction and normalization, but inspect results.

## 4. Example (high level)
```python
from foodspec.preprocess.ftir import AtmosphericCorrector, SimpleATRCorrector
from foodspec.preprocess.raman import CosmicRayRemover

atm = AtmosphericCorrector()
atr = SimpleATRCorrector()
cr = CosmicRayRemover()

X_ft = atm.transform(X_ft)
X_ft = atr.transform(X_ft, wavenumbers=wn)
X_ra = cr.transform(X_ra)
```

## 5. Visuals to include
- **FTIR atmospheric correction:** Single FTIR spectrum before/after water/CO₂ subtraction; annotate removed bands. Axes: wavenumber vs intensity. Use `AtmosphericCorrector` + `plot_spectra`.
- **Raman cosmic ray removal:** Raw Raman spectrum with a spike and cleaned version (spike replaced); mark the removed spike. Axes: wavenumber vs intensity. Use `CosmicRayRemover` + `plot_spectra`.

## Reproducible figure generation
- Use a helper such as `docs/examples/visualization/generate_scatter_cosmic_figures.py` to create figures for this chapter:
  - Build a synthetic Raman spectrum with one or two sharp spikes; apply `CosmicRayRemover` and overlay before/after (save to `docs/assets/cosmic_ray_cleanup.png`).
  - Take an FTIR spectrum with broad water/CO₂ bands (example oils FTIR or synthetic); apply `AtmosphericCorrector` (and optional `SimpleATRCorrector`) and overlay before/after with annotations (save to `docs/assets/ftir_atmospheric_correction.png`).
  - Optionally show a short PCA scatter of raw vs corrected FTIR spectra to illustrate reduced variance from atmospheric artefacts.

## Summary
- Scatter and atmospheric effects distort baselines/intensities; use SNV/MSC plus targeted corrections.
- Cosmic ray spikes in Raman should be removed to avoid biasing normalization/peaks.
- Always validate corrections visually.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for scatter correction and cosmic ray removal:**

1. **Cosmic rays not removed from spectra (single-pixel spikes affect normalization and averaging)**
   - Cosmic ray (high intensity noise) inflates normalization factors
   - Averaged spectra have artifacts
   - **Fix:** Detect cosmic rays (statistical outliers per wavelength); interpolate or remove; validate visually

2. **Scatter correction method not validated on reference (using MSC without checking it reduces scatter)**
   - Some scatter-correction methods ineffective for specific scatter types
   - Scatter may remain post-correction
   - **Fix:** Apply reference-free QC (compare spectra before/after); validate on samples with known scatter levels

3. **Atmospheric water/CO₂ lines not masked in FTIR (strong H₂O/CO₂ peaks affect ratios)**
   - Water/CO₂ lines dominate certain regions; ratios computed in these regions are meaningless
   - Information loss
   - **Fix:** Mask atmospheric bands before feature extraction; list masked regions; validate peaks not in masked regions

4. **Multiplicative scatter correction (MSC) applied assuming linear reference-sample relationship (breaks if sample composition fundamentally different)**
   - MSC assumes sample and reference vary only in scale/offset
   - Non-linear scatter or complex matrix breaks assumption
   - **Fix:** Validate MSC on test samples; check residuals post-MSC; consider Extended MSC or nonlinear alternatives

5. **Spikes/cosmic rays interpolated without validation (interpolated values treated as real data)**
   - Interpolation introduces artificial spectra
   - May create false peaks or alter ratios
   - **Fix:** Mark interpolated regions; avoid extracting features from interpolated wavelengths; use robust statistics

6. **Scatter not uniform across samples (one sample highly scattering, another clear, corrected with single MSC)**
   - Sample-specific scatter variation not captured by single MSC
   - Residual scatter remains
   - **Fix:** Apply sample-specific scatter correction; check scatter by visual inspection; group similar scatter types

7. **Removal of high-frequency noise (cosmic rays) also removes true high-frequency information**
   - Aggressive spike removal flattens sharp true peaks
   - Information loss for peak-based features
   - **Fix:** Use conservative spike detection (>5 SD from local mean); visualize removed vs. original; preserve sharp features

8. **No validation of scatter correction on known samples (assuming correction works without testing)**
   - Different sample types may respond differently to scatter correction
   - Overcorrection or undercorrection undetected
   - **Fix:** Test scatter correction on reference materials; compare corrected/uncorrected downstream metrics

## Further reading
- [Baseline correction](../baseline_correction/)
- [Normalization & smoothing](../normalization_smoothing/)
- [PCA and dimensionality reduction](../ml/pca_and_dimensionality_reduction.md)
