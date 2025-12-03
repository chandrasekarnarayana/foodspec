# Preprocessing: Normalization, Smoothing, and Scatter Correction

Normalization and smoothing reduce variability caused by intensity scaling, path length, and noise. This chapter explains how to stabilize spectra before feature extraction.

## 1. Why normalize and smooth?
- **Intensity scaling:** Laser power fluctuations, ATR contact pressure, and sample thickness change absolute intensities.
- **Scatter:** Mie/Rayleigh scatter and refractive-index changes reshape baselines and peak heights.
- **Noise:** Electronic noise, shot noise; high-resolution spectra can be noisy.

## 2. Smoothing and derivatives
### Savitzky–Golay
- Polynomial fit in a moving window; preserves peak shape better than boxcar averaging.
- Parameters: `window_length`, `polyorder`; larger windows smooth more but risk peak distortion.
- Derivatives: 1st/2nd derivatives enhance peak contrast and separate overlapping bands.

### Moving average
- Simple and fast; good for mild noise; may broaden peaks.

**When to use:** Apply light smoothing before peak detection; use derivatives when peaks overlap or baseline is hard to model. Avoid over-smoothing small/weak peaks.

## 3. Normalization & scatter correction
### Vector / area / max normalization
- Scale each spectrum to unit norm, unit area, or unit max. Good for compensating intensity differences.
- Pitfalls: Sensitive to outliers or noise spikes for max scaling.

### Internal-peak normalization
- Scale by intensity or area of a stable reference band (e.g., phenyl ring mode). Requires known stable band.

### SNV (Standard Normal Variate)
- Per spectrum: subtract mean, divide by standard deviation. Reduces scatter-induced scaling.
- Pitfalls: Distorts absolute intensities; avoid when absolute areas are needed.

### MSC (Multiplicative Scatter Correction)
- Fits each spectrum to a reference (often the mean spectrum): \( x_{\text{corr}} = (x - a)/b \).
- Good for contact/path-length differences; assumes linear scatter effects.

## 4. Cropping and windowing
- Use cropping to exclude noisy edges, atmospheric regions, or irrelevant bands.
- Typical crops: Raman fingerprint (600–1800 cm⁻¹), CH stretches (2800–3100 cm⁻¹); FTIR users may exclude strong water/CO₂ regions if not corrected.

## 5. Modality-specific helpers (preview)
- **ATR/Atmospheric correction:** See [Scatter & cosmic-ray removal](scatter_correction_cosmic_ray_removal.md).
- **Cosmic rays:** Raman spike removal to prevent distortion of normalization metrics.

## 6. Example (high level)
```python
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import SNVNormalizer

sg = SavitzkyGolaySmoother(window_length=11, polyorder=3)
snv = SNVNormalizer()
X_smooth = sg.transform(X_raw)
X_norm = snv.transform(X_smooth)
```

## 7. Visuals to include
- Figure: Raw vs smoothed vs derivative spectrum.
- Figure: Effects of vector vs SNV vs MSC on a set of replicate spectra.

## Summary
- Normalize to stabilize intensity scaling; smooth to reduce noise while preserving peaks.
- Choose SNV/MSC for scatter-heavy data; internal-peak/area for stable reference bands.
- Crop irrelevant regions to improve robustness.

## Further reading
- [Baseline correction](baseline_correction.md)
- [Scatter & cosmic-ray handling](scatter_correction_cosmic_ray_removal.md)
- [Feature extraction](feature_extraction.md)
- [Classification & regression](../ml/classification_regression.md)
