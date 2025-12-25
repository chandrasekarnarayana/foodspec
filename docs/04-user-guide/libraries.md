# Libraries & Public Datasets

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Users converting raw spectral files to FoodSpec's HDF5 format; researchers accessing public datasets.
**What problem does this solve?** Understanding FoodSpec's data storage format and how to create/load spectral libraries.
**When to use this?** When you have raw spectral data (CSV/TXT files) and need to create a reusable library for workflows.
**Why it matters?** Proper data organization in HDF5 format enables reproducible analysis, faster loading, and metadata tracking.
**Time to complete:** 10 minutes to read; library creation takes seconds to minutes depending on dataset size.
**Prerequisites:** Raw spectral data files; FoodSpec installed; basic understanding of data formats

---

Questions this page answers
- What is stored in a FoodSpec HDF5 library?
- How do I build/load libraries?
- Which public datasets are used in the protocol?
- How does CSV → FoodSpectrumSet → HDF5 work?

## What is stored in an HDF5 library?
- `x`: spectra matrix (n_samples × n_wavenumbers)
- `wavenumbers`: shared axis (cm⁻¹)
- `metadata`: one row per sample (labels, acquisition info)
- `modality`: Raman / FTIR / NIR tag
- `provenance`: basic version/timestamp metadata

## Building and loading libraries
```python
from pathlib import Path
from foodspec.io import create_library
from foodspec import load_library

input_folder = Path("data/oils_raw")
out_path = Path("libraries/oils_raman.h5")
create_library(input_folder, out_path, modality="raman")
fs = load_library(out_path)
```

## Public datasets (used in protocol examples)

| Dataset | Modality | Task | Approx size | DOI/URL |
| --- | --- | --- | --- | --- |
| Mendeley edible oils | Raman/FTIR | Classification | Tens–hundreds spectra | Mendeley Data (edible oils) |
| EVOO–sunflower mixtures | Raman | Regression (fraction) | Tens spectra | DOI: 10.57745/DOGT0E |
| FTIR edible oils (ATR) | FTIR | Classification/adulteration | Tens spectra | Public FTIR oil datasets |

Loaders: `load_public_mendeley_oils`, `load_public_evoo_sunflower_raman`, `load_public_ftir_oils` (data must be pre-downloaded).

## CSV → FoodSpectrumSet → HDF5
- Use `foodspec csv-to-library` for CSV inputs.
- **Wide**: one column per spectrum, `wavenumber` column for the axis.
- **Long/tidy**: rows = (sample_id, wavenumber, intensity), extra metadata columns allowed.
- Command:
```bash
foodspec csv-to-library data/oils.csv libraries/oils.h5 \
  --format wide \
  --wavenumber-column wavenumber \
  --label-column oil_type \
  --modality raman
```
- Internally: CSV → FoodSpectrumSet (validated) → HDF5 with spectra, axis, metadata, modality, provenance.

## Public loader examples
```python
from foodspec.data import load_public_mendeley_oils, load_public_evoo_sunflower_raman
fs_cls = load_public_mendeley_oils(root="path/to/mendeley")
fs_mix = load_public_evoo_sunflower_raman(root="path/to/evoo_sunflower")
```
These return validated FoodSpectrumSet objects ready for workflows.

See also
- [csv_to_library.md](csv_to_library.md)
- [workflows/oil_authentication.md](../workflows/oil_authentication.md)
- [workflows/mixture_analysis.md](../workflows/mixture_analysis.md)
