# Preprocessing: Feature Extraction (Peaks, Bands, Ratios, Fingerprints)

Feature extraction turns preprocessed spectra into quantitative descriptors for chemometrics and ML. This chapter covers peaks, band integration, ratios, and fingerprint similarity.

## 1. Peaks
- **What:** Locate local maxima near expected wavenumbers; measure height and area.
- **Why:** Specific functional groups (e.g., C=C at ~1655 cm⁻¹, C=O at ~1742 cm⁻¹) are diagnostic for oils; amide bands for proteins.
- **How:** Provide `expected_peaks` and `tolerance` to capture the nearest maximum; compute height and optionally area in a window.
- **Pitfalls:** Overlapping peaks; noisy derivatives; baseline errors can bias heights—ensure preprocessing is solid.

## 2. Band integration
- **What:** Integrate intensity over defined ranges to capture broader features.
- **Why:** Useful for broad NIR bands or overlapping FTIR features; robust to small shifts.
- **How:** Define `(label, min_wn, max_wn)`; integrate numerically.
- **Pitfalls:** Include baseline if not corrected; window choice matters.

## 3. Ratios
- **What:** Ratios of peak heights/areas or band integrals (e.g., 1655/1742 for unsaturation vs carbonyl).
- **Why:** Normalize out scaling, emphasize relative chemistry; common in authenticity and degradation studies.
- **How:** Specify numerator/denominator feature names; handle zeros/NaNs carefully.
- **Pitfalls:** Ratios amplify noise when denominators are small; ensure peaks are present.

## 4. Fingerprint similarity
- **What:** Cosine/correlation similarity between spectra or feature vectors.
- **Why:** Library search, QC thresholds, clustering; supports novelty detection or matching to references.
- **How:** Use cosine/correlation matrices; interpret high similarity as close matches.
- **Pitfalls:** Sensitive to preprocessing differences; ensure consistent normalization and cropping.

## 5. Example (high level)
```python
from foodspec.features.peaks import PeakFeatureExtractor
from foodspec.features.ratios import RatioFeatureGenerator

extractor = PeakFeatureExtractor(expected_peaks=[1655.0, 1742.0], tolerance=8.0)
peak_feats = extractor.fit_transform(X_proc, wavenumbers=wn)
ratio_gen = RatioFeatureGenerator({"ratio_1655_1742": ("peak_1655.0_height", "peak_1742.0_height")})
ratios = ratio_gen.transform(peak_feats)
```

## 6. Visuals to include
- Annotated spectrum showing expected peaks and integration windows.
- Histogram of a key ratio by class (oil_type) or time point.
- Similarity matrix heatmap for library search/QC.

## Summary
- Peaks capture localized functional groups; bands capture broader features; ratios emphasize relative chemistry; fingerprints support similarity/QC tasks.
- Good preprocessing (baseline, normalization) is critical before feature extraction.
- Choose features to match the scientific question: authentication (peaks/ratios), mixtures (areas, similarity), degradation (time-trend ratios).

## Further reading
- [Baseline correction](baseline_correction.md)
- [Normalization & smoothing](normalization_smoothing.md)
- [Classification & regression](../ml/classification_regression.md)
- [Mixture analysis workflow](../workflows/mixture_analysis.md)
