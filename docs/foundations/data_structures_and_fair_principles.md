# Foundations: Data Structures and FAIR Principles

This chapter explains how FoodSpec represents spectral data and how to keep analyses FAIR (Findable, Accessible, Interoperable, Reusable). It introduces the core data models and storage formats used throughout the book.

## 1. Core data models
- **FoodSpectrumSet:** 2D array `x` (n_samples × n_wavenumbers), shared `wavenumbers` (1D, ascending cm⁻¹), `metadata` (pandas DataFrame), `modality` tag (`"raman"`, `"ftir"`, `"nir"`).
- **HyperSpectralCube:** 3D array `(height, width, n_wavenumbers)` with optional flattening to a FoodSpectrumSet for pixel-wise analysis.
- **Validation:** Monotonic axes, matching shapes, metadata length equals n_samples; see [validation utilities](../03-cookbook/validation_chemometrics_oils.md) and `foodspec.validation`.

## 2. Storage formats
- **HDF5 libraries:** Preferred for reproducibility; store `x`, `wavenumbers`, `metadata_json`, `modality`, provenance (software version, timestamps). See [Libraries](../04-user-guide/libraries.md).
- **CSV (wide/long):** Common export from instruments; convert to HDF5 via [CSV → HDF5 pipeline](../04-user-guide/csv_to_library.md).
- **Provenance:** Keep config files, run metadata, model registry entries; see [Reproducibility checklist](../protocols/reproducibility_checklist.md).

## 3. FAIR principles applied
- **Findable:** Clear file names, metadata columns (sample_id, label columns like oil_type), DOI/URLs for public datasets.
- **Accessible:** Use open formats (CSV, HDF5) and documented folder structures.
- **Interoperable:** Monotonic wavenumbers in cm⁻¹, standard column names, modality tags; avoid vendor lock-in.
- **Reusable:** Record preprocessing choices, model configs, seeds, software versions; archive reports and model artifacts.

## 4. When to use which structure
- **Batch analyses:** FoodSpectrumSet for single-spot spectra; choose HDF5 libraries for storage and sharing.
- **Imaging:** HyperSpectralCube for spatial maps; flatten to FoodSpectrumSet for pixel-wise ML, then reshape labels/maps.
- **Library search/QC:** Maintain curated HDF5 libraries with consistent metadata; use fingerprint similarity or one-class models.

## 5. Example (high level)
```python
from foodspec.core.dataset import FoodSpectrumSet
from foodspec.data.libraries import create_library, load_library

# Build in memory
fs = FoodSpectrumSet(x=..., wavenumbers=..., metadata=..., modality="raman")

# Persist to HDF5
create_library(path="libraries/oils.h5", spectra=fs.x, wavenumbers=fs.wavenumbers,
               metadata=fs.metadata, modality=fs.modality)
fs_loaded = load_library("libraries/oils.h5")
```

## Summary
- FoodSpectrumSet and HyperSpectralCube are the backbone of analyses.
- Use HDF5 with provenance for FAIR, reproducible storage.
- Standardize axes, metadata, and modality to stay interoperable.

## Further reading
- [CSV → HDF5 pipeline](../04-user-guide/csv_to_library.md)
- [Libraries & public datasets](../04-user-guide/libraries.md)
- [Reproducibility checklist](../protocols/reproducibility_checklist.md)
- [API hub](../api/index.md)

---

## When Results Cannot Be Trusted

⚠️ **Red flags for data structures and FAIR compliance:**

1. **Metadata missing or incomplete (files named sample1.csv with no sample ID, date, instrument info)**
   - Cannot reproduce analysis; provenance lost
   - Data not reusable or findable
   - **Fix:** Include sample ID, date, instrument, operator, prep protocol in metadata; use structured formats (HDF5, CSV with metadata header)

2. **File format changes mid-project (CSV for batch 1, Excel for batch 2)**
   - Inconsistent parsing; data cleaning errors
   - Analysis scripts break
   - **Fix:** Freeze data format before project start; convert all to common format (HDF5, CSV) with validation

3. **No versioning or changelog (data files overwritten; no record of changes)**
   - Cannot trace data evolution; reproducibility lost
   - Errors undetectable
   - **Fix:** Use version control (Git); document changes in CHANGELOG; never overwrite raw data

4. **Raw data not archived (only processed data saved)**
   - Cannot reprocess if preprocessing errors found
   - Can't apply new methods
   - **Fix:** Archive raw data separately; document processing steps; keep processing scripts with data

5. **Data not FAIR (stored locally, not shared, no DOI, no license)**
   - Not findable, accessible, interoperable, or reusable
   - Scientific reproducibility impossible
   - **Fix:** Deposit data in public repository (Zenodo, Figshare, domain-specific); assign DOI; add CC-BY or CC0 license

6. **Data structure incompatible with FoodSpec (wrong column names, missing wavenumber axis)**
   - FoodSpec expects specific data structure
   - Parsing failures or silent errors
   - **Fix:** Follow FoodSpec data spec; use validation tools; test import before analysis
