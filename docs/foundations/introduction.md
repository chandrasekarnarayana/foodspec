# Foundations: Introduction

FoodSpec is a headless, research-grade toolkit for Raman, FTIR, and NIR spectroscopy in food science. These docs are written as a **textbook + protocol manual**: you can read linearly to learn the physics and computation, or jump to workflows and API examples for immediate use.

## What this chapter covers
- Why FoodSpec is documented as a book (to be teachable, citable, and reproducible).
- Who the intended readers are (spectroscopists, chemists, physicists, data/ML scientists, QC engineers).
- How to navigate Parts I–VI and the appendices.
- The data and metadata assumptions baked into the library.

## How to use this book
1. **If you are new to vibrational spectroscopy:** Start with Part I (Foundations), especially [Spectroscopy basics](spectroscopy_basics.md), then skim Part II on preprocessing.
2. **If you are an ML/DS practitioner:** Skim Part I for units/conventions, focus on Parts II–III, then jump to workflows in Part IV.
3. **If you need to ship analyses:** Go directly to Part IV workflows (oil, heating, mixture, QC, hyperspectral) and Part V for reproducibility/benchmarking.
4. **If you need API details:** Use Part VI (API hub) and the keyword index.

### Data assumptions (for all chapters)
- Spectra are arrays indexed by wavenumber (cm⁻¹) in **ascending order**.
- Metadata is tabular (sample_id, labels like oil_type, process conditions like heating_time).
- Preferred storage is HDF5 with provenance; see [CSV → HDF5 pipeline](../csv_to_library.md) and [Libraries](../libraries.md).

### Typical learning pathway
- Front matter quickstarts → Foundations → Preprocessing → ML/Chemometrics → Workflows → Protocols/Benchmarks → API.
- Use the sidebar as a table of contents; “See also” links connect related concepts (e.g., PCA → Classification).

## Summary
- FoodSpec is documented as a structured book to support both learning and rigorous protocol use.
- Read linearly to build understanding, or jump to workflows and API for immediate tasks.
- Keep data in cm⁻¹, ascending order, with clear metadata and provenance.

## Further reading
- [Spectroscopy basics](spectroscopy_basics.md)
- [Libraries & public datasets](../libraries.md)
- [Baseline correction](../preprocessing/baseline_correction.md)
- [PCA and dimensionality reduction](../ml/pca_and_dimensionality_reduction.md)
