# CSV → HDF5 pipeline

foodspec converts CSVs to reusable spectral libraries (HDF5) for all workflows.

## Supported inputs
| Format | Description | Required columns | Typical use |
| --- | --- | --- | --- |
| CSV (wide) | One column per spectrum, one row per wavenumber | `wavenumber`, sample columns | Fast conversion to HDF5 |
| CSV (long/tidy) | One row per (sample_id, wavenumber, intensity) | `sample_id`, `wavenumber`, `intensity` | Public datasets, tidy data |
| Folder of TXT/CSV | One file per spectrum, aligned axes | filename, wavenumber/intensity columns | Instrument exports |
| HDF5 library | Serialized FoodSpectrumSet | x, wavenumbers, metadata, modality | Primary format for workflows |

## CSV layouts
### Wide format
```
wavenumber,s1,s2,s3
500,10.1,12.2,9.9
502,10.3,12.4,10.0
```
- wavenumber is the axis; each other column is a spectrum.

### Long/tidy format
```
sample_id,wavenumber,intensity,oil_type
s001,500,10.1,olive
s001,502,10.3,olive
s002,500,12.2,sunflower
```
- one row per (sample, wavenumber); extra columns become metadata (e.g., oil_type).

## CLI: csv-to-library
```bash
foodspec csv-to-library \
  data/oils.csv \
  libraries/oils.h5 \
  --format wide \
  --wavenumber-column wavenumber \
  --modality raman \
  --label-column oil_type
```
- Use `--format long` with `--sample-id-column` / `--intensity-column` for tidy data.
- Parent directories are created automatically; output is an HDF5 library.

## Python API
```python
from foodspec.io.csv_import import load_csv_spectra
from foodspec.io import create_library

fs = load_csv_spectra("data/oils.csv", format="wide", modality="raman")
create_library("libraries/oils.h5", spectra=fs.x, wavenumbers=fs.wavenumbers, metadata=fs.metadata, modality=fs.modality)
```

## Troubleshooting
- Missing column errors: check spelling of wavenumber/sample_id/intensity.
- Non-monotonic axis: sort wavenumbers before conversion.
- Mixed labels: ensure consistent sample IDs and metadata rows.
