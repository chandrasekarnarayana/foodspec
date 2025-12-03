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
- See [Normalization & smoothing](normalization_smoothing.md); SNV/MSC mitigate path-length/contact effects via linear rescaling to a reference.

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

## Further reading
- [Baseline correction](baseline_correction.md)
- [Normalization & smoothing](normalization_smoothing.md)
- [PCA and dimensionality reduction](../ml/pca_and_dimensionality_reduction.md)
