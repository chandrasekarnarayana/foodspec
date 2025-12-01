# Data formats & libraries

## Core object: FoodSpectrumSet
- `x`: array (n_samples × n_wavenumbers)
- `wavenumbers`: 1D axis (cm⁻¹)
- `metadata`: pandas DataFrame with one row per sample
- `modality`: Raman / FTIR / NIR

## Supported inputs
| Format | Description | Required columns | Typical use |
| --- | --- | --- | --- |
| CSV (wide) | One column per spectrum, one row per wavenumber | `wavenumber`, sample columns | Fast conversion to HDF5 |
| CSV (long/tidy) | One row per (sample_id, wavenumber, intensity) | `sample_id`, `wavenumber`, `intensity` | Public datasets, tidy data |
| Folder of TXT/CSV | One file per spectrum, aligned axes | filename, wavenumber/intensity columns | Instrument exports |
| HDF5 library | Serialized FoodSpectrumSet | x, wavenumbers, metadata, modality | Primary format for workflows |

## Libraries (HDF5)
- Created via `foodspec csv-to-library` (CLI) or `create_library` (Python).
- Store spectra, axis, metadata JSON, modality, and provenance.
- Validated for shape and monotonic wavenumbers.
