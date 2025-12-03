# Preprocessing: Normalization, Smoothing, and Scatter Correction

## Why this matters in food spectroscopy
- Intensity scaling varies with laser power, ATR contact, and sample thickness.
- Scatter (Mie/Rayleigh) and refractive-index changes reshape baselines/peaks.
- Noise (electronic/shot) masks weak bands; smoothing stabilizes peak finding.

## When to use these methods
- **Smoothing/derivatives:** Before peak detection; when noise obscures peaks or peaks overlap.
- **Normalization/scatter correction:** When comparing spectra across batches/instruments; when scatter or path-length differences bias peak heights.
- **When not to:** Avoid heavy smoothing on very weak/short peaks; avoid SNV/MSC if absolute intensities/areas must be preserved without scaling.

## How it works (short version)
- **Savitzky–Golay:** Local polynomial fit in a moving window; derivatives enhance peak contrast.
- **Moving average:** Simple boxcar averaging; broadens peaks.
- **Vector/area/max normalization:** Rescales each spectrum to unit norm/area/max.
- **Internal-peak normalization:** Rescales by a stable reference band.
- **SNV (Standard Normal Variate):** Per spectrum \( x' = (x - \bar{x})/s_x \); reduces multiplicative scatter.
- **MSC (Multiplicative Scatter Correction):** Fit \( x \approx a + b \cdot x_{\text{ref}} \); output \( (x-a)/b \); handles linear scatter/path-length effects.

## Practical guidance and pitfalls
- Choose `window_length`/`polyorder` cautiously; larger windows smooth more but risk peak distortion.
- SNV/MSC help when replicate spectra differ in slope/scale; may distort absolute areas.
- Max normalization is sensitive to spikes; prefer area/vector for robustness.
- Crop noisy/irrelevant regions (e.g., Raman fingerprint 600–1800 cm⁻¹; CH stretches 2800–3100 cm⁻¹).

## Minimal Python example
```python
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import SNVNormalizer

sg = SavitzkyGolaySmoother(window_length=11, polyorder=3)
snv = SNVNormalizer()
X_smooth = sg.transform(X_raw)
X_norm = snv.transform(X_smooth)
```

## Typical plots and figures
- Raw vs smoothed vs derivative spectrum (0th/1st/2nd).
- Overlays showing effects of vector vs SNV vs MSC on replicates.
- Cropped vs full-range spectra.

## Further reading / see also
- [Baseline correction](baseline_correction.md)
- [Scatter & cosmic-ray handling](scatter_correction_cosmic_ray_removal.md)
- [Feature extraction](feature_extraction.md)
- [Classification & regression](../ml/classification_regression.md)
