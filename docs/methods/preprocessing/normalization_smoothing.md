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

---

## When Results Cannot Be Trusted

⚠️ **Red flags for normalization and smoothing validity:**

1. **Normalization method not matched to data characteristics (vector normalization on spectra with very different magnitudes)**
   - Vector norm emphasizes high-intensity regions; may not be appropriate for all data
   - SNV or MSC better if multiplicative effects present
   - **Fix:** Test multiple normalizations; choose based on data characteristics; validate downstream metrics

2. **Smoothing window too large (7-point Savitzky-Goyal on sharp peaks 5 points wide)**
   - Large smoothing windows flatten true narrow peaks
   - Information loss for peak-based features
   - **Fix:** Ensure smoothing window smaller than narrowest true feature; use conservative window_length

3. **Smoothing and derivatives without examining polynomial order (2nd-order polyorder removes true 2nd-derivative features)**
   - Polynomial order affects what's preserved; high order fits noise
   - Derivatives amplify noise; high polynomial order makes worse
   - **Fix:** Use low polyorder (2–3); visualize smoothed/derivative spectra; validate peaks preserved

4. **Normalization parameters (SNV mean/SD) computed on all data, then applied to test set**
   - Data leakage: test set statistics influence training
   - Proper workflow: compute normalization on training data only
   - **Fix:** Include normalization in preprocessing pipeline; compute statistics only on training data

5. **Different normalization methods applied to spectra from different sources (Device A vector-normalized, Device B SNV)**
   - Spectra not comparable; downstream models fail
   - Batch effects introduced by normalization
   - **Fix:** Apply identical normalization to all data; freeze parameters before analysis

6. **Cropping wavenumber range without justification (removing low-wavenumber noise regions containing informative peaks)**
   - Important chemical information removed
   - Chemometric models trained on partial spectra won't generalize
   - **Fix:** Document crop regions; validate that no informative peaks removed; test models on uncropped data

7. **Over-smoothing (window_length = 201 on 1800-point spectra) makes features disappear**
   - Excessively smoothed spectra lose all peak structure
   - Ratios collapse; information completely lost
   - **Fix:** Use conservative smoothing (window_length ≤10 points); visualize smoothed spectra; verify peaks visible

8. **Normalization makes zero/negative intensities (spectra divided by zero or normalized to zero mean, producing negative values)**
   - Invalid for log-transforms, ratio calculations
   - Suggests over-normalization or data issue
   - **Fix:** Check raw spectra before normalization; ensure no zero values; use robust normalizations (MSC with outlier handling)

## When to Use

Use normalization when:

- **Inter-sample comparison**: Spectra collected under varying conditions (different laser powers, ATR contact pressures, path lengths)
- **Multiplicative scatter effects**: Raman spectra show variable baseline slopes from particle size differences
- **Batch effects**: Spectra from different instruments or sessions need harmonization
- **Classification/clustering**: Relative peak patterns matter more than absolute intensities
- **Reference standardization**: Creating spectral libraries that must be instrument-independent

Use smoothing when:

- **Noisy spectra**: Electronic or shot noise obscures weak peaks
- **Peak detection**: Preparing for automated peak finding algorithms
- **Derivative calculations**: Computing 1st or 2nd derivatives that amplify noise
- **Visual clarity**: Presenting spectra in figures or reports

## When NOT to Use (Common Failure Modes)

Avoid normalization when:

- **Absolute intensities are meaningful**: Concentration measurements require preserving intensity scales
- **Zero/negative values present**: Many normalizations fail with non-positive data
- **Very sparse spectra**: Spectra with few peaks where normalization distorts relative heights
- **After ratio calculation**: Normalizing ratios can invalidate calibration models
- **Extreme outliers present**: SNV/vector norm amplify outlier effects before removal

Avoid smoothing when:

- **Sharp peaks are critical**: Narrow Raman bands (FWHM < 5 cm⁻¹) where smoothing causes peak broadening
- **Already low noise**: Over-smoothing removes true spectral fine structure
- **Before baseline correction**: Smoothing can complicate baseline identification
- **Quantitative peak fitting**: Fitting Lorentzian/Gaussian profiles to smoothed data introduces bias

## Recommended Defaults

**For Savitzky-Golay smoothing:**
```python
from foodspec.preprocessing import savgol_smooth
X_smoothed = savgol_smooth(X, window_length=11, polyorder=3, deriv=0)
```
- `window_length=11`: Balances noise reduction and peak preservation
- `polyorder=3`: Cubic fit preserves peak shapes
- `deriv=0`: Direct smoothing; use `deriv=1` or `deriv=2` for derivatives

**For SNV normalization (scatter correction):**
```python
from foodspec.preprocessing import normalize_snv
X_normalized = normalize_snv(X)
```
- No parameters; centers and scales each spectrum independently

**For vector normalization (unit L2 norm):**
```python
from foodspec.preprocessing import VectorNormalizer
normalizer = VectorNormalizer()
X_normalized = normalizer.fit_transform(X)
```
- Preserves relative peak heights; reduces intensity scale differences

**For MSC (multiplicative scatter correction):**
```python
from foodspec.preprocessing import MSCNormalizer
msc = MSCNormalizer()
X_corrected = msc.fit_transform(X)
```
- Fit on training data; removes linear scatter effects relative to mean spectrum

## See Also

**API Reference:**

- [savgol_smooth](../../api/preprocessing.md) - Savitzky-Golay filter
- [normalize_snv](../../api/preprocessing.md) - Standard Normal Variate
- [VectorNormalizer](../../api/preprocessing.md) - L2 normalization
- [MSCNormalizer](../../api/preprocessing.md) - Multiplicative scatter correction

**Related Methods:**

- [Baseline Correction](baseline_correction.md) - Apply before normalization
- [Scatter Correction & Cosmic Ray Removal](scatter_correction_cosmic_ray_removal.md) - Complementary preprocessing
- [Derivatives & Feature Enhancement](derivatives_and_feature_enhancement.md) - Advanced smoothing applications

**Examples:**

- [Preprocessing Pipeline](../../examples_gallery.md) - Complete normalization workflows
- [Quality Control](../../examples_gallery.md) - Normalization for batch comparison

## Further reading / see also
- [Baseline correction](baseline_correction.md)
- [Scatter & cosmic-ray handling](scatter_correction_cosmic_ray_removal.md)
- [Feature extraction](feature_extraction.md)
- [Classification & regression](../chemometrics/classification_regression.md)
