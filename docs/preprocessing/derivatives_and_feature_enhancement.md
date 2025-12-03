# Preprocessing: Derivatives and Feature Enhancement

Derivatives enhance subtle or overlapping peaks and help remove slowly varying backgrounds. This chapter explains when to compute derivatives and how to balance sensitivity with noise.

## 1. Why derivatives?
- **Peak separation:** 1st/2nd derivatives sharpen overlapping bands (common in lipid/protein mixtures).
- **Baseline suppression:** Low-order derivatives reduce slow baseline trends, complementing explicit baseline correction.
- **Feature sensitivity:** Derivatives emphasize changes in slope; useful for subtle adulteration signals.

## 2. Methods
- **Savitzky–Golay derivatives:** Fit local polynomials and differentiate analytically; preserves peak shape better than finite differences.
- **Finite differences:** Simple but noisier; usually avoided when Savitzky–Golay is available.

Parameters to tune:
- `window_length` and `polyorder`: choose larger windows for smoother derivatives; ensure `window_length` > `polyorder` and odd.
- Derivative order: 1st derivative for slope/peak edges; 2nd derivative for peak sharpening and resolving overlaps.

## 3. When to use / not to use
- **Use when:** Baseline is hard to model; peaks overlap; you need subtle discriminative features (e.g., small adulterant signals).
- **Avoid when:** Very low SNR (derivatives amplify noise); peaks are already well separated; absolute intensities are needed for quantitation.

## 4. Example (high level)
```python
from foodspec.preprocess.derivatives import DerivativeTransformer

deriv = DerivativeTransformer(order=1, window_length=11, polyorder=3)
X_d1 = deriv.transform(X_preprocessed)
```
Preview derivatives alongside raw spectra with `foodspec.viz.plot_spectra` or a custom overlay.

## 5. Visuals to include
- **Derivative overlays (single spectrum):** Plot original, 1st, and 2nd derivatives for a preprocessed oil spectrum (e.g., example oils). Axes: wavenumber vs intensity/derivative units. Purpose: show peak sharpening vs noise. Use `DerivativeTransformer` + `plot_spectra` for overlays.
- **Overlap/utility check:** Apply PCA on raw vs derivative spectra (same dataset) and compare score plots; illustrates discriminative gain. Axes: PC1 vs PC2 colored by class.

## Reproducible figure generation
- Figures for this chapter can be generated with `docs/examples/visualization/generate_derivative_overlays.py`, which should:
  - Load the example oils dataset.
  - Plot original, 1st-, and 2nd-derivative spectra for a representative sample and save to `docs/assets/derivative_overlays.png`.
  - Build PCA on raw vs 1st-derivative spectra and plot PC1–PC2 colored by oil type to compare separation; save to `docs/assets/derivative_pca_comparison.png`.
  - Use `DerivativeTransformer`, `SavitzkyGolaySmoother`, and the plotting helpers in `foodspec.viz` for consistent styling.

## Summary
- Derivatives can separate overlapping peaks and suppress slow baselines.
- Savitzky–Golay derivatives balance smoothing and differentiation; tune window/polyorder carefully.
- Use with caution in noisy regimes; consider combining with mild smoothing first.

## Further reading
- [Normalization & smoothing](normalization_smoothing.md)
- [Feature extraction](feature_extraction.md)
- [PCA and dimensionality reduction](../ml/pca_and_dimensionality_reduction.md)
