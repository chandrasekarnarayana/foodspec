# Foundations: Data Structures and FAIR Principles

This chapter explains how FoodSpec represents spectral data and how to keep analyses FAIR (Findable, Accessible, Interoperable, Reusable). It introduces the core data models and storage formats used throughout the book.

## 1. Core data models
- **FoodSpectrumSet:** 2D array `x` (n_samples × n_wavenumbers), shared `wavenumbers` (1D, ascending cm⁻¹), `metadata` (pandas DataFrame), `modality` tag (`"raman"`, `"ftir"`, `"nir"`).
- **HyperSpectralCube:** 3D array `(height, width, n_wavenumbers)` with optional flattening to a FoodSpectrumSet for pixel-wise analysis.
- **Validation:** Monotonic axes, matching shapes, metadata length equals n_samples; see [validation utilities](../validation_chemometrics_oils.md) and `foodspec.validation`.

## 2. Storage formats
- **HDF5 libraries:** Preferred for reproducibility; store `x`, `wavenumbers`, `metadata_json`, `modality`, provenance (software version, timestamps). See [Libraries](../libraries.md).
- **CSV (wide/long):** Common export from instruments; convert to HDF5 via [CSV → HDF5 pipeline](../csv_to_library.md).
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
- [CSV → HDF5 pipeline](../csv_to_library.md)
- [Libraries & public datasets](../libraries.md)
- [Reproducibility checklist](../protocols/reproducibility_checklist.md)
- [API hub](../api/index.md)
