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
```

## Troubleshooting
- **Unsupported format**: ensure extension matches table; otherwise convert to CSV/JCAMP.
- **Missing dependency**: install the appropriate extra (`spc`, `opus`); ImportError messages guide installation.
- **Wavenumber issues**: verify axis is ascending cm⁻¹; flip/order if needed before analysis.
- **Sparse metadata**: filenames become `sample_id`; vendor headers may provide instrument/date; add missing metadata manually if needed.

## See also
- [Workflow design](../workflows/workflow_design_and_reporting.md)
- [Libraries & public datasets](../libraries.md)
- [CSV → HDF5 pipeline](../csv_to_library.md)
- [API: IO & data](../api/io.md)
