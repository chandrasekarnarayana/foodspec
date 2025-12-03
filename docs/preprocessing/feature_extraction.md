# Preprocessing: Feature Extraction (Peaks, Bands, Ratios, Fingerprints)

Feature extraction turns preprocessed spectra into quantitative descriptors for chemometrics and ML. This chapter covers peaks, band integration, ratios, and fingerprint similarity.

## Why this matters in food spectroscopy
- Peaks and bands correspond to chemical groups; ratios and integrals summarize composition changes (e.g., unsaturation vs carbonyl, oxidation markers).
- Quantitative features feed PCA/PLS/ML; poor feature choices limit interpretability and performance.

## 1. Peaks
- **What:** Locate local maxima near expected wavenumbers; measure height and area.
- **Why:** Specific functional groups (e.g., C=C at ~1655 cm⁻¹, C=O at ~1742 cm⁻¹) are diagnostic for oils; amide bands for proteins.
- **How (user-defined):** You specify the peaks you care about via `expected_peaks` and a `tolerance` (window, e.g., ±6–10 cm⁻¹). The extractor searches only within that window to find the nearest local maximum, which accounts for small shifts from calibration, temperature, matrix effects, or preprocessing. Heights/areas are then computed in that window.
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

### Peak and ratio choices for food spectroscopy
- **Chemical meaning first:** anchor peaks/bands to known vibrational assignments from literature.
- **Illustrative oil examples (Raman):**
  - Peaks: ~1655 cm⁻¹ (C=C stretch, unsaturation), ~1440 cm⁻¹ (CH₂ bend), ~1265 cm⁻¹ (cis =C–H bend), ~717 cm⁻¹ (C–C stretch).
  - Ratios: 1655/1440 (unsaturation vs saturation), 1655/717 (degree of unsaturation), 1265/1440 (cis content vs saturated backbone). Higher unsaturation ratios generally indicate more double bonds.
- **Identification:** detect local maxima near user-specified positions ± tolerance (e.g., 6–10 cm⁻¹) to accommodate small shifts, or integrate fixed windows (e.g., 1650–1665 cm⁻¹) for robustness. Users control the expected peaks and tolerances; the extractor confines the search to that window.
- **Interpretation:** track ratio changes across classes or treatments (e.g., heating decreases 1655/1440 as unsaturation drops). Use either the example oils dataset or a synthetic Raman spectrum from `generate_synthetic_raman_spectrum` to illustrate the labeled peaks and ratios.

### Assessing peak stability across spectra/groups
- **Goal:** quantify how stable peak positions and intensities are across replicates or groups (e.g., batches, treatments).
- **Procedure:** run `PeakFeatureExtractor` on all spectra; compute mean/SD (or CV) of detected peak positions and intensities per group using pandas; optionally run ANOVA/Kruskal on intensities to test group differences.
- **Interpretation:** low SD in position suggests good spectral alignment; low CV in intensity implies stable measurement/preprocessing; large shifts or CV may indicate instrument drift, poor preprocessing, or real chemical changes.
- **Tools:** combine peak outputs with stats utilities (e.g., `run_anova`, `run_kruskal_wallis`) for intensity comparison across groups.

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
![Annotated peaks and band](../assets/peak_band_annotation.png)

*Figure: Synthetic spectrum with peaks marked and a shaded band area for integration (e.g., ratio or area feature).*

- Histogram of a key ratio by class (oil_type) or time point.
- Similarity matrix heatmap for library search/QC.

## How to choose bands and ratios (decision mini-guide)
> Guidance, not a rigid rule. Always validate against your matrix and literature.

**Common questions → suggested bands/ratios**
- **Distinguishing oils/fat composition:**
  - Bands: ~1655 cm⁻¹ (C=C stretch, unsaturation), ~1742 cm⁻¹ (C=O ester), ~1450 cm⁻¹ (CH₂ bend).
  - Ratios: 1655/1742 (unsaturation), 1450/1655 (saturation vs unsaturation balance).
  - Why: Unsaturation and carbonyl content differ by oil type. See [Oil authentication](../workflows/oil_authentication.md).
- **Tracking oxidation/degradation (heating):**
  - Bands: decrease at ~1655 cm⁻¹ (C=C), increase near ~1710–1740 cm⁻¹ (oxidation carbonyls).
  - Ratios: 1655/1740 (unsaturation loss), 1450/1655 (relative saturation), optional 3010/2850 (CH stretch region if available).
  - Why: Unsaturation decreases; carbonyls grow with oxidation. See [Heating quality monitoring](../workflows/heating_quality_monitoring.md).
- **Adulteration with known adulterant:**
  - Bands: choose unique marker of adulterant (e.g., specific carbonyl or aromatic band).
  - Ratios: marker band / stable reference band (e.g., 1742/1655 if adulterant rich in carbonyl).
  - Why: Highlight component unique to adulterant; validate against pure references.
- **Batch QC comparison:**
  - Bands: stable backbone bands (e.g., 1450 cm⁻¹ CH₂) vs marker band of interest.
  - Ratios: marker/reference to detect drift; use control limits.
  - Why: Quick screening for drift/outliers in production.

**Code sketch (ratios + boxplot)**
```python
import pandas as pd
import seaborn as sns
from foodspec.data.loader import load_example_oils
from foodspec.features.ratios import compute_ratios

fs = load_example_oils()
# assumes peak heights already extracted into metadata/features
ratio_def = {"unsat_ratio": ("peak_1655.0_height", "peak_1742.0_height")}
ratio_df = compute_ratios(fs.metadata, ratio_def)
ratio_df["oil_type"] = fs.metadata["oil_type"].values
sns.boxplot(data=ratio_df, x="oil_type", y="unsat_ratio")
```
*How to read:* Higher unsat_ratio suggests more unsaturation; between-group differences can be tested with ANOVA/Kruskal and effect sizes (Cohen’s d). Pair the plot with p-values/effect sizes for publication.

## Summarizing Peaks and Ratios Across Groups
- **Peak stats:** mean ± std of peak position and intensity indicate alignment/stability (position) and consistency/variation (intensity). Shifts may reflect chemistry (e.g., oxidation) or preprocessing/instrument drift.
- **Ratio tables:** mean ± std (and n) of key ratios by group (oil_type, batch, treatment) give a compact view for reports/supplementary tables.

### Example (peak stats + ratio tables)
```python
import pandas as pd
from foodspec.features import PeakFeatureExtractor, RatioFeatureGenerator, compute_peak_stats, compute_ratio_table
from foodspec.data.loader import load_example_oils

fs = load_example_oils()
extractor = PeakFeatureExtractor(expected_peaks=[1655.0, 1742.0], tolerance=8.0)
peak_df = extractor.fit_transform(fs.x, wavenumbers=fs.wavenumbers)
peak_df["spectrum_id"] = fs.metadata.index

ratio_gen = RatioFeatureGenerator({"ratio_1655_1742": ("peak_1655.0_height", "peak_1742.0_height")})
ratio_df = ratio_gen.transform(peak_df)

peak_summary = compute_peak_stats(peak_df, metadata=fs.metadata, group_keys=["oil_type"])
ratio_summary = compute_ratio_table(ratio_df.join(fs.metadata[["oil_type"]]), metadata=fs.metadata, group_keys=["oil_type"])
print(peak_summary.head())
print(ratio_summary.head())
```
*Interpretation:* A band with different mean positions across groups may indicate chemical/environmental shifts; large intensity SD within group suggests variability/noise. Ratio tables with clear between-group differences but small within-group std support discriminative power; pair with ANOVA/Kruskal and effect sizes.

**Cross-links**
- Vibrational assignments: [Spectroscopy basics](../foundations/spectroscopy_basics.md#vibrational-modes-and-their-spectral-signatures).
- Workflows using these ratios: [Oil authentication](../workflows/oil_authentication.md), [Heating quality monitoring](../workflows/heating_quality_monitoring.md).
- Chemometric context: [Chemometrics guide](../chemometrics_guide.md).

## Summary
- Peaks capture localized functional groups; bands capture broader features; ratios emphasize relative chemistry; fingerprints support similarity/QC tasks.
- Good preprocessing (baseline, normalization) is critical before feature extraction.
- Choose features to match the scientific question: authentication (peaks/ratios), mixtures (areas, similarity), degradation (time-trend ratios).

## Further reading
- [Baseline correction](baseline_correction.md)
- [Normalization & smoothing](normalization_smoothing.md)
- [Vibrational modes and spectral signatures](../foundations/spectroscopy_basics.md)
- [Classification & regression](../ml/classification_regression.md)
- [Mixture analysis workflow](../workflows/mixture_analysis.md)
