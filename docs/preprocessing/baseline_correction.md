# Preprocessing: Baseline Correction

Baselines drift because of fluorescence, scattering, ATR contact, and instrument response. Poor baselines distort peak heights/areas and ratios. This chapter explains when and how to correct baselines in Raman/FTIR/NIR spectra.

> For notation and symbols used below, see the [Glossary](../glossary.md).

## 1. Why baselines drift
- **Fluorescence (Raman):** Broad background that can dwarf weak peaks.
- **Scattering/contact (FTIR ATR):** Sloping backgrounds from poor contact or refractive-index mismatches.
- **Instrument response:** Slowly varying offsets/gains.
- **Sample matrix:** Particulates, emulsions, and path-length changes introduce curvature.

## 2. When (not) to correct
- **Correct if:** Baseline dominates dynamic range; ratios/areas are biased; spectra share similar baseline shape.
- **Caution if:** Peaks are broad and might be mistaken for baseline; very low SNR; automated correction can remove real signal.
- **Visual check:** Always inspect before/after; consider preserving a copy of raw data.

## 3. Methods in FoodSpec
### 3.1 Asymmetric Least Squares (ALS)
> **Math box (ALS objective)**  
> Minimize \( \sum_i w_i (y_i - b_i)^2 + \lambda \sum_i (\Delta^2 b_i)^2 \)  
> with asymmetric weights \( w_i \) that down-weight peaks (controlled by \( p \)).
- **Concept:** Fit a smooth baseline \( b \) minimizing the above objective with asymmetric weights to downweight peaks.
- **Parameters:** `lambda_` (smoothness), `p` (asymmetry), `max_iter`.
- **When to use:** Moderate to strong curvature; mixed peaks/baseline; widely used default.
- **Pitfalls:** Over-smoothing if `lambda_` too large; peak clipping if `p` too small.

### 3.2 Rubberband (Convex Hull)
- **Concept:** Compute lower convex hull of the spectrum and interpolate baseline.
- **When to use:** Spectra with clear gaps between peaks and background; quick and parameter-light.
- **Pitfalls:** Can fail if peaks are dense or baseline is above hull; sensitive to noise spikes.

### 3.3 Polynomial baseline
- **Concept:** Fit low-degree polynomial to spectrum (or selected background regions).
- **When to use:** Mild curvature, simple backgrounds.
- **Pitfalls:** Overfitting high-degree polynomials; unsuitable for complex fluorescence.

## 4. Practical guidance
- **Order in pipeline:** Baseline → smoothing/normalization → features. Avoid applying after aggressive normalization.
- **Quality checks:** Plot before/after; monitor peak heights/areas; compare across replicates.
- **Parameter tuning:** Start with moderate `lambda_` (e.g., 1e5) and `p` (~0.001–0.01) for ALS; adjust visually.
- If over/under-correction persists, see [Common problems & solutions](../troubleshooting/common_problems_and_solutions.md#c-preprocessing--chemometric-problems).

## 5. Example (high level)
```python
from foodspec.preprocess.baseline import ALSBaseline

als = ALSBaseline(lambda_=1e5, p=0.001, max_iter=10)
X_corr = als.transform(X_raw)
```

## 6. Visuals to include
- **Baseline before/after (synthetic):** One spectrum with quadratic drift + noise. Show raw (with drift), true baseline, and corrected signal (e.g., subtract known baseline). Axes: wavenumber (cm⁻¹) vs intensity. Purpose: illustrate what a successful correction looks like. See `docs/examples/visualization/generate_baseline_before_after.py`.

![Baseline before/after](../assets/baseline_before_after.png)

- **Drift/noise illustration:** Overlay ideal spectrum, noisy spectrum, and drifted spectrum to show why correction is needed. Axes: wavenumber vs intensity. Generated via `docs/examples/stats/generate_spectral_artifacts_figures.py`.

![Illustrative spectra with baseline drift and noise](../assets/spectra_artifacts.png)

## Reproducible figure generation
- Run `python docs/examples/visualization/generate_baseline_before_after.py` for baseline before/after.
- Run `python docs/examples/stats/generate_spectral_artifacts_figures.py` for drift/noise illustration.

## Summary
- Baseline correction removes broad backgrounds that bias peak-based features.
- ALS is a flexible default; rubberband is simple; polynomial fits suit mild curvature.
- Always inspect corrections; avoid removing true broad features.

## Further reading
- [Normalization & smoothing](normalization_smoothing.md)
- [Feature extraction](feature_extraction.md)
- [Raman/FTIR practical guide](../ftir_raman_preprocessing.md)
- [PCA](../ml/pca_and_dimensionality_reduction.md)
