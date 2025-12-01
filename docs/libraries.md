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

## CSV → FoodSpectrumSet → HDF5

You can convert standalone CSV files into a `FoodSpectrumSet` and then into an HDF5 library:

```python
from pathlib import Path
from foodspec.io.csv_import import load_csv_spectra
from foodspec.data.libraries import create_library

csv_path = Path("data/public_oils_wide.csv")
fs = load_csv_spectra(csv_path, format="wide", wavenumber_column="wavenumber")

# Save as HDF5 library for downstream workflows
create_library("libraries/public_oils.h5", fs)
```

Supported CSV formats:
- **wide**: one wavenumber column and one column per spectrum (rows = wavenumbers).
- **long**: rows are (sample_id, wavenumber, intensity); use `format="long"` and set the column names if different.

After import, the resulting HDF5 can be used directly with CLI commands (e.g., `foodspec oil-auth`) or Python workflows.

## Public datasets and CSV-based libraries

In the MethodsX protocol and examples, foodspec does **not** ship data.
Instead, we rely on **public, citable datasets** that you download yourself
and convert into `FoodSpectrumSet` HDF5 libraries.

### Datasets used in the protocol

We currently demonstrate the protocol on three open datasets:

1. **Raman and Infrared Spectroscopic Analysis: Classification of Edible Oils**  
   *Modality*: Raman, MIR, NIR  
   *Task*: multi-class edible oil classification; PCA and chemometrics.  
   *Source*: Mendeley Data, dataset “Raman and Infrared Spectroscopic Analysis: Classification of Edible Oils”.

2. **Raman spectra of extra virgin olive and sunflower oil mixtures**  
   *Modality*: Raman (500–2500 cm⁻¹)  
   *Task*: regression of EVOO fraction in EVOO–sunflower mixtures; mixture analysis.  
   *Source*: French national data portal (data.gouv.fr), DOI **10.57745/DOGT0E**.

3. **Groundnut Oil Adulteration (ATR-MIR)**  
   *Modality*: ATR-MIR  
   *Task*: detection and quantification of palm oil adulteration in groundnut oil.  
   *Source*: Kaggle dataset “Groundnut Oil Adulteration”.

> See the MethodsX paper and `docs/methodsx_protocol.md` for how these map
> to the protocol figures and tables.

### Supported CSV formats

`foodspec` can convert CSV files into HDF5 libraries via the
`foodspec csv-to-library` command. Two CSV layouts are supported:

#### 1. Wide format

One row per wavenumber, **one column per spectrum**:

```text
wavenumber,sample_001,sample_002,sample_003
500,       123.4,     98.1,      110.2
502,       124.0,     99.2,      111.0
...
```

- `wavenumber` is the x-axis (cm⁻¹).
- Each other column is a spectrum for one sample.
- Optional labels (e.g. `oil_type`) can be supplied by a separate metadata CSV and merged before conversion.

Example command:

```bash
foodspec csv-to-library \
  data/public/mendeley_oils_raman.csv \
  libraries/mendeley_oils_raman.h5 \
  --format wide \
  --wavenumber-column wavenumber \
  --modality raman \
  --label-column oil_type
```

#### 2. Long / tidy format

One row per (sample_id, wavenumber):

```text
sample_id,wavenumber,intensity,oil_type
s001,     500,       123.4,    olive
s001,     502,       124.0,    olive
s002,     500,       98.1,     sunflower
...
```

- `sample_id` identifies the spectrum.
- `wavenumber` is the x-axis (cm⁻¹).
- `intensity` is the signal value.
- Any extra column (e.g. `oil_type`, `evoo_fraction`) becomes metadata.

Example command:

```bash
foodspec csv-to-library \
  data/public/evoo_sunflower_long.csv \
  libraries/evoo_sunflower_raman.h5 \
  --format long \
  --sample-id-column sample_id \
  --wavenumber-column wavenumber \
  --intensity-column intensity \
  --label-column evoo_fraction \
  --modality raman
```

Internally, `csv-to-library`:

- Loads the CSV → `FoodSpectrumSet`.
- Validates shapes, monotonic wavenumbers, NaNs, etc. (via `validate_spectrum_set`).
- Saves an HDF5 file (`x`, `wavenumbers`, `metadata_json`, `modality`), which can be
  consumed by any foodspec workflow (`oil-auth`, `heating`, `protocol-benchmarks`,
  `reproduce-methodsx`, etc.).
