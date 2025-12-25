# Foundations: Introduction

FoodSpec is a headless, research-grade toolkit for Raman, FTIR, and NIR spectroscopy in food science. These docs are written as a **textbook + protocol manual**: you can read linearly to learn the physics and computation, or jump to workflows and API examples for immediate use.

## What this chapter covers
- Why FoodSpec is documented as a book (to be teachable, citable, and reproducible).
- Who the intended readers are (spectroscopists, chemists, physicists, data/ML scientists, QC engineers).
- How to navigate Parts I–VI and the appendices.
- The data and metadata assumptions baked into the library.

## Mathematical background & prerequisites
- **Linear algebra:** covariance matrices, eigenvalues/eigenvectors (for PCA/PLS), vector norms.
- **Statistics:** basic probability, p-values, confidence intervals, effect sizes.
- **Signals:** smoothing/derivatives, baseline drift notions.
- The docs keep math light; every concept has code examples and defaults. You can follow workflows without deep math; cross-links point to theory sections for more detail.

See also:
- PCA/PLS math sketch in [PCA & dimensionality reduction](../ml/pca_and_dimensionality_reduction.md)
- Baseline/normalization math boxes in [Preprocessing](../preprocessing/baseline_correction.md) and [Normalization & smoothing](../preprocessing/normalization_smoothing.md)
- Stats assumptions and tests in [Hypothesis testing](../stats/hypothesis_testing_in_food_spectroscopy.md) and [Nonparametric methods](../stats/nonparametric_methods_and_robustness.md)

## How to use this book
1. **If you are new to vibrational spectroscopy:** Start with Part I (Foundations), especially [Spectroscopy basics](spectroscopy_basics.md), then skim Part II on preprocessing.
2. **If you are an ML/DS practitioner:** Skim Part I for units/conventions, focus on Parts II–III, then jump to workflows in Part IV.
3. **If you need to ship analyses:** Go directly to Part IV workflows (oil, heating, mixture, QC, hyperspectral) and Part V for reproducibility/benchmarking.
4. **If you need API details:** Use Part VI (API hub) and the keyword index.

### Data assumptions (for all chapters)
- Spectra are arrays indexed by wavenumber (cm⁻¹) in **ascending order**.
- Metadata is tabular (sample_id, labels like oil_type, process conditions like heating_time).
- Preferred storage is HDF5 with provenance; see [CSV → HDF5 pipeline](../04-user-guide/csv_to_library.md) and [Libraries](../04-user-guide/libraries.md).

### Typical learning pathway
- Front matter quickstarts → Foundations → Preprocessing → ML/Chemometrics → Workflows → Protocols/Benchmarks → API.
- Use the sidebar as a table of contents; “See also” links connect related concepts (e.g., PCA → Classification).

## Summary
- FoodSpec is documented as a structured book to support both learning and rigorous protocol use.
- Read linearly to build understanding, or jump to workflows and API for immediate tasks.
- Keep data in cm⁻¹, ascending order, with clear metadata and provenance.

## Further reading
- [Spectroscopy basics](spectroscopy_basics.md)
- [Libraries & public datasets](../04-user-guide/libraries.md)
- [Baseline correction](../preprocessing/baseline_correction.md)
- [PCA and dimensionality reduction](../ml/pca_and_dimensionality_reduction.md)

---

## When Results Cannot Be Trusted

⚠️ **Red flags for spectroscopy data fundamentals:**

1. **Raw spectra not inspected before analysis (assuming preprocessing fixed everything)**
   - Hidden artifacts (cosmic rays, baseline shifts, fluorescence) can bias downstream analysis
   - Preprocessing cannot fix fundamental data quality issues
   - **Fix:** Always visualize raw spectra; check for artifacts; document raw data quality

2. **Sample preparation not documented (assuming spectra reflect sample composition without stating prep protocol)**
   - Different prep (dilution, grinding, solvent) dramatically affects spectra
   - Comparisons across preps invalid
   - **Fix:** Document all sample prep steps; standardize protocols; report prep in methods

3. **Instrument settings not recorded (unknown laser power, integration time, grating)**
   - Irreproducible measurements
   - Cannot troubleshoot issues or harmonize across instruments
   - **Fix:** Record all instrument settings; include in metadata; document for every measurement

4. **No quality control standards (no blanks, no reference materials, no positive controls)**
   - Unknown if instrument/method working correctly
   - Drift or contamination undetected
   - **Fix:** Include blanks every session; measure reference materials; track QC over time

5. **Data versioning/metadata lost (files renamed, original metadata stripped)**
   - Cannot trace samples to acquisition conditions
   - Reproducibility impossible
   - **Fix:** Preserve original filenames; maintain metadata; use structured data formats (HDF5, CSV with metadata)

6. **Multiple spectroscopy methods compared without validation (Raman vs FTIR data analyzed together)**
   - Different methods, different spectral signatures
   - Models don't transfer without retraining
   - **Fix:** Analyze methods separately; validate cross-method transfer explicitly
