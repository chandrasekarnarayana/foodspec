# Instrument & File Formats Guide

FoodSpec normalizes instrument exports into a common representation (`FoodSpectrumSet` / HDF5 libraries). Vendor formats are input routes; analysis always operates on the normalized form.

## Supported formats (overview)

| Format type    | Extension(s)              | How to load                         | Extra dependency?        |
| -------------- | ------------------------- | ----------------------------------- | ------------------------ |
| CSV (wide)     | `.csv`                    | `read_spectra("file.csv")`          | No                       |
| CSV (folder)   | `.csv` in a directory     | `read_spectra("folder/")`           | No                       |
| JCAMP-DX       | `.jdx`, `.dx`             | `read_spectra("file.jdx")`          | No (built-in parser)     |
| SPC            | `.spc`                    | `read_spectra("file.spc")`          | Yes (`pip install foodspec[spc]`) |
| Bruker OPUS    | `.0`, `.1`, `.opus`       | `read_spectra("file.0")`            | Yes (`pip install foodspec[opus]`) |
| TXT            | `.txt`                    | `read_spectra("file.txt")`          | No |

## Typical structure and metadata
- **Spectral axis**: wavenumber (cm⁻¹), ascending, 1D.
- **Intensity**: arbitrary units; one column per spectrum (wide CSV) or one file per spectrum (folder/JCAMP/vendor).
- **Metadata**: sample_id (from filename/column), plus any vendor header info (instrument, date) when available.
- **Normalization**: vendor loaders return raw intensities; downstream preprocessing handles baselines/normalization.
- **Coverage**: document spectral range/resolution; ensure exported range includes target bands (fingerprint, CH stretch).

## Examples
```python
from foodspec.io import read_spectra

# CSV (wide)
fs = read_spectra("data/oils_wide.csv")

# Folder of instrument CSV exports
fs_folder = read_spectra("data/export_folder/")

# JCAMP-DX
fs_jdx = read_spectra("data/sample.jdx")

# SPC (requires optional extra)
# pip install foodspec[spc]
fs_spc = read_spectra("data/sample.spc")

# OPUS (requires optional extra)
# pip install foodspec[opus]
fs_opus = read_spectra("data/sample.0")

# Run a quick PCA to verify structure
from foodspec.chemometrics.pca import run_pca
pca, res = run_pca(fs_opus.x, n_components=2)
print(res.explained_variance_ratio_)

# After ingest, run any standard workflow (e.g., oil authentication)
# from foodspec.apps.oils import run_oil_authentication_workflow
# result = run_oil_authentication_workflow(fs_opus, label_column=\"oil_type\")
```

### Notes on formats and quirks
- **CSV (wide/folder):** Sometimes missing units; ensure wavenumber (cm⁻¹) and ascending axis. Folder exports often store one spectrum per file; filenames become `sample_id`.
- **JCAMP-DX (.jdx/.dx):** Multi-block files may contain multiple spectra; FoodSpec reads blocks into separate spectra. Check headers for units (wavenumber vs wavelength); we assume cm⁻¹ and convert when obvious.
- **SPC (.spc):** Binary; may contain multiple traces. Requires `pip install foodspec[spc]`. If missing, you’ll see `ImportError: SPC support requires the 'spc' extra`; install the extra to proceed.
- **OPUS (.0/.1/.opus):** Binary Bruker format; may contain multiple spectra. Requires `pip install foodspec[opus]`. Missing dependency raises `ImportError: OPUS support requires the 'opus' extra`.
- **TXT:** Treat as CSV-like; ensure delimiter and column names; wavenumber column required.

All vendor formats are normalized to the same internal `FoodSpectrumSet` representation (x, wavenumbers, metadata). Downstream workflows are format-agnostic once loaded.

### Example: SPC (commercial) → FoodSpectrumSet → HDF5
```python
from foodspec.io import read_spectra
from foodspec.data.libraries import create_library

# Requires: pip install foodspec[spc]
fs_spc = read_spectra("data/vendor/sample.spc")
create_library(
    spectra=fs_spc.x,
    wavenumbers=fs_spc.wavenumbers,
    metadata=fs_spc.metadata,
    modality=fs_spc.modality,
    path="libraries/sample_spc.h5",
)
```
If the extra is missing, you’ll see an ImportError mentioning the `spc` extra—install it and rerun.

### Example: OPUS (commercial) → FoodSpectrumSet → HDF5
```python
# Requires: pip install foodspec[opus]
fs_opus = read_spectra("data/vendor/sample.0")
create_library(
    spectra=fs_opus.x,
    wavenumbers=fs_opus.wavenumbers,
    metadata=fs_opus.metadata,
    modality=fs_opus.modality,
    path="libraries/sample_opus.h5",
)
```
If the extra is missing, an ImportError will suggest installing the `opus` extra.

### Synthetic vendor overlay
![Synthetic vendor overlay](../assets/vendor_overlay.png)

Synthetic spectra (mimicking SPC/OPUS after normalization) overlaid to illustrate that vendor imports are reduced to the standard wavenumber/intensity layout before analysis.

## Troubleshooting
- **Unsupported format**: ensure extension matches table; otherwise convert to CSV/JCAMP.
- **Missing dependency**: install the appropriate extra (`spc`, `opus`); ImportError messages guide installation.
- **Wavenumber issues**: verify axis is ascending cm⁻¹; flip/order if needed before analysis.
- **Sparse metadata**: filenames become `sample_id`; vendor headers may provide instrument/date; add missing metadata manually if needed.

## See also
- [Workflow design](../workflows/workflow_design_and_reporting.md)
- [Oil authentication workflow](../workflows/authentication/oil_authentication.md) for an end-to-end run after vendor ingest
- [Libraries & public datasets](../user-guide/libraries.md)
- [CSV → HDF5 pipeline](../user-guide/csv_to_library.md)
- [API: IO & data](../api/io.md)
