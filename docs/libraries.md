# Libraries and public datasets

foodspec provides helpers to build and load spectral libraries from folders of
files, as well as convenience loaders for several public edible-oil datasets.

## Building a library

You can convert spectra into an HDF5 library:

```python
from pathlib import Path
from foodspec.data.loader import load_example_oils
from foodspec.data.libraries import create_library, load_library

ds = load_example_oils()
library_path = Path("libraries/oils_raman.h5")
create_library(library_path, ds)
fs = load_library(library_path)
```

The library stores spectra (`x`), wavenumbers, metadata, modality, and basic provenance.

## Public dataset loaders

Several loaders are provided for public edible-oil datasets. These functions do
not download data automatically; they expect that you have downloaded and placed
the files in a documented folder structure.

### Mendeley edible-oil dataset

```python
from foodspec.data import load_public_mendeley_oils
fs_mend = load_public_mendeley_oils(root="path/to/mendeley_oils")
```

`root` should point to the directory containing the dataset CSV files. Metadata includes
`oil_type`, dataset/source fields, and DOI placeholders.

### EVOO–sunflower mixture dataset

```python
from foodspec.data import load_public_evoo_sunflower_raman
fs_mix = load_public_evoo_sunflower_raman(root="path/to/evoo_sunflower")
```

Metadata includes `mixture_fraction_evoo` and related fields. `validate_public_evoo_sunflower`
is applied to check mixture fractions are in valid ranges.

### FTIR edible-oil dataset

```python
from foodspec.data import load_public_ftir_oils
fs_ftir = load_public_ftir_oils(root="path/to/ftir_oils")
```

Returns a `FoodSpectrumSet` with `modality="ftir"` and oil-type metadata.

All public loaders:

- return a validated `FoodSpectrumSet`,
- call `validate_spectrum_set` to check wavenumber ordering and metadata shape,
- raise clear errors if files are missing or in unexpected formats.

### Expected folder structures (fill in with actual DOIs/links)

- **Mendeley oils**: `~/foodspec_datasets/mendeley_oils/*.csv`
- **EVOO–sunflower Raman**: `~/foodspec_datasets/evoo_sunflower_raman/*.csv`
- **FTIR oils**: `~/foodspec_datasets/ftir_oils/*.csv`

Each CSV should have wavenumbers as the first column and intensities as subsequent columns
(or one spectrum per file with two columns).
