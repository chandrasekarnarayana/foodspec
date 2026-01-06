# User Guide – Data Formats & HDF5

**Purpose:** Understand how to format your data (CSV vs HDF5) and choose the right format for reproducibility.

**Audience:** Lab managers preparing datasets; researchers building reproducible pipelines.

**Time:** 15–20 minutes to choose format; 10 min per dataset to validate.

**Prerequisites:** Familiarity with CSV files or HDF5 basics.

---

This page explains supported input formats, the HDF5 layout, and vendor IO expectations.

**Why it matters:** Choosing the right format affects reproducibility (FAIR metadata), harmonization, and ease of loading via CLI.

## CSV vs HDF5: Quick Comparison

| Feature | CSV | HDF5 |
|---------|-----|------|
| Learning curve | Easy | Medium |
| File size | Large | Compact |
| Metadata | Limited | Full (FAIR) |
| Speed (1000+ samples) | Slow | Fast |
| Preprocessing history | Not stored | Stored |
| Multi-instrument data | Awkward | Natural |
| **Best for** | Getting started | Production pipelines |

**Example: Convert CSV → HDF5**

```python
from foodspec.io import load_csv, to_hdf5

# Load CSV
ds = load_csv("oils.csv", wavenumber_col="wavenumber")

# Convert to HDF5 with metadata
to_hdf5(
    ds,
    "oils_processed.h5",
    instrument_metadata={"laser": 785, "grating": "1200/mm"},
    preprocessing_log=["baseline_als(lam=1e6)", "normalize_snv()"]
)
```

## Choosing Your Format

**Use CSV if:**
- ✅ Dataset < 500 samples
- ✅ Single instrument, single batch
- ✅ Quick exploratory analysis
- ✅ Sharing via email/GitHub

**Use HDF5 if:**
- ✅ Dataset > 500 samples
- ✅ Multiple instruments or batches
- ✅ Need preprocessing/protocol history
- ✅ Publishing reproducible research
- ✅ Integrating 3D hyperspectral cubes

## CSV vs HDF5
- **CSV**: Wide-format with wavenumber columns and metadata columns (oil_type, matrix, heating_stage, replicate, batch, etc.). Easiest to start with.
- **HDF5**: Preferred for FAIR storage. FoodSpec uses a NeXus-inspired layout with explicit groups and units.

## HDF5 layout (simplified)
- `/spectra/wn_axis`: wavenumber axis (units attr: `cm^-1`)
- `/spectra/intensities`: spectra matrix (n_samples × n_wavenumbers)
- `/spectra/sample_table`: annotations (oil_type, matrix, heating_stage, batch, replicate, instrument, etc.)
- `/instrument/`: laser_wavelength_nm, grating, objective, calibration parameters
- `/preprocessing/`: list of preprocessing steps with parameters
- `/protocol/`: protocol name/version, step definitions, validation strategy
- Attributes: `foodspec_hdf5_schema_version` for compatibility

Notes:
- HDF5 retains preprocessing/protocol history, visible in metadata.

## Vendor IO
- FoodSpec supports generic CSV/HDF5 and provides vendor loader stubs (OPUS/WiRE/ENVI). If binary parsing is incomplete, export to CSV or HDF5 from your instrument software.
- Plugins can register additional vendor loaders; see `registry_and_plugins.md`.
- Error messages will hint at missing blocks/headers if a vendor file is malformed; follow the suggested export path (e.g., “export as ASCII/CSV”).

## Choosing a format
- Use **CSV** for quick starts and small datasets.
- Use **HDF5** for multi-instrument/batch projects, HSI cubes, and when you want provenance and harmonization metadata preserved.
- For HSI, store cubes and segmentation outputs in HDF5; label maps and ROI tables are also written to run bundles.

## Mini-workflow
1) Export data as CSV (wide) or HDF5 using FoodSpec save functions.  
2) Load via CLI (`--input my.h5`).  
3) Run a protocol; verify `metadata.json` reflects format, preprocessing, harmonization.

## Next Steps

- **Data loading:** [Loading spectra from files](../api/io.md)
- **Preprocessing:** [Baseline correction and smoothing](../methods/preprocessing/baseline_correction.md)
- **Vendor integration:** [Vendor I/O guide](vendor_io.md)
- **Reference:** [Data format schema](../reference/data_format.md)

See also: [cookbook_preprocessing.md](../methods/preprocessing/normalization_smoothing.md) and [registry_and_plugins.md](registry_and_plugins.md) for vendor plugins.
