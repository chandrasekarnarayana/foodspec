# User Guide – Data formats & HDF5

This page explains supported input formats, the HDF5 layout, and vendor IO expectations.

**Why it matters:** Choosing the right format affects reproducibility (FAIR metadata), harmonization, and ease of loading in GUI/CLI.

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

GUI notes:
- The **Data & Mapping** panel reads CSV/HDF5; CSV requires numeric wavenumber columns, HDF5 auto-loads axes/sample table.
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
2) Load in GUI (Data & Mapping) or CLI (`--input my.h5`).  
3) Run a protocol; verify `metadata.json` reflects format, preprocessing, harmonization.

See also: [cookbook_preprocessing.md](../03-cookbook/cookbook_preprocessing.md) and [registry_and_plugins.md](registry_and_plugins.md) for vendor plugins.
