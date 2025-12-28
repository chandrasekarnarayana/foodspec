# IO & Data API Reference

!!! info "Module Purpose"
    Loading, saving, and converting spectral data from/to various formats (CSV, HDF5, JCAMP-DX, vendor formats).

---

## Quick Navigation

| Function | Purpose | Common Use |
|----------|---------|------------|
| [`load_folder()`](#load_folder) | Load spectra from directory | Batch import text files |
| [`load_from_metadata_table()`](#load_from_metadata_table) | Load from metadata CSV | Structured datasets |
| [`read_jcamp()`](#vendor-formats) | Read JCAMP-DX files | FTIR/Raman vendor data |
| [`read_opus()`](#vendor-formats) | Read Bruker OPUS | Bruker FTIR instruments |
| [`to_hdf5()`](#exporters) | Save to HDF5 | Efficient storage |
| [`to_csv()`](#exporters) | Export to CSV | Excel/pandas compatibility |

---

## Common Patterns

### Pattern 1: Load Folder of Text Files

```python
from foodspec.io import load_folder

# Load all .txt files from directory
fs = load_folder(
    folder="data/oils/raw/",
    pattern="*.txt",
    modality="raman",
    metadata_csv="data/oils/metadata.csv",  # Optional: merge metadata by sample_id
    wavenumber_column=0,  # First column = wavenumbers
    intensity_columns=None  # All other columns = intensities (averaged)
)
print(f"Loaded: {len(fs)} spectra")
print(f"Wavenumber range: {fs.wavenumbers[0]:.1f}-{fs.wavenumbers[-1]:.1f} cm⁻¹")
```

### Pattern 2: Load with Metadata Merging

```python
# metadata.csv:
# sample_id,oil_type,batch,supplier
# spectrum_001,Olive,A,SupplierX
# spectrum_002,Sunflower,A,SupplierY

fs = load_folder(
    "data/oils/",
    pattern="*.txt",
    metadata_csv="data/oils/metadata.csv"
)

# Metadata automatically merged by sample_id (file stem)
print(fs.metadata[['sample_id', 'oil_type', 'batch']].head())
```

### Pattern 3: Save/Load HDF5 for Efficiency

```python
from foodspec.io import to_hdf5, from_hdf5

# Save to HDF5 (compact, fast)
to_hdf5(fs, "data/oils_processed.h5", compression='gzip')
print(f"Saved: {len(fs)} spectra to HDF5")

# Load from HDF5
fs_loaded = from_hdf5("data/oils_processed.h5")
assert len(fs_loaded) == len(fs)
assert fs_loaded.hash() == fs.hash()
```

### Pattern 4: Export to CSV for Excel

```python
from foodspec.io import to_csv

# Export spectra + metadata to CSV
to_csv(
    fs,
    output_path="data/oils_export.csv",
    include_metadata=True,
    wavenumber_header_prefix="wn_"
)
# CSV columns: sample_id, oil_type, batch, wn_600.0, wn_601.0, ...
```

---

## Loaders

### load_folder

Load multiple spectra from a folder of text files.

::: foodspec.io.loaders.load_folder
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.io import load_folder

# Basic usage
fs = load_folder(
    folder="data/raman_spectra/",
    pattern="sample_*.txt",
    modality="raman"
)

# With metadata merging
fs = load_folder(
    folder="data/raman_spectra/",
    pattern="*.txt",
    metadata_csv="data/metadata.csv",  # Must have 'sample_id' column
    wavenumber_column=0,
    intensity_columns=[1, 2, 3]  # Average columns 1-3
)
print(f"Loaded: {len(fs)} spectra with {fs.metadata.columns.tolist()} metadata")
```

**Parameters:**
- `folder`: Directory containing spectra files
- `pattern`: Glob pattern (e.g., `"*.txt"`, `"sample_*.csv"`)
- `modality`: `"raman"`, `"ftir"`, or `"nir"`
- `metadata_csv`: Optional CSV with `sample_id` column for metadata merging
- `wavenumber_column`: Index of wavenumber column (default: 0)
- `intensity_columns`: Indices to average (default: all except wavenumber column)

**Returns:** `FoodSpectrumSet` with common wavenumber axis

**See Also:** [`load_from_metadata_table()`](#load_from_metadata_table)

---

### load_from_metadata_table

Load spectra listed in a metadata table with file paths.

::: foodspec.io.loaders.load_from_metadata_table
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.io import load_from_metadata_table

# metadata.csv:
# file_path,oil_type,batch,replicate
# data/oil_001.txt,Olive,A,1
# data/oil_002.txt,Olive,A,2
# data/oil_003.txt,Sunflower,B,1

fs = load_from_metadata_table(
    metadata_csv="data/metadata.csv",
    modality="raman"
)
print(f"Loaded: {len(fs)} spectra")
print(f"Metadata columns: {fs.metadata.columns.tolist()}")
```

**Parameters:**
- `metadata_csv`: CSV with `file_path` column + optional metadata columns
- `modality`: Spectroscopy modality
- `wavenumber_column`: Index of wavenumber column in spectrum files
- `intensity_columns`: Indices to average

**Returns:** `FoodSpectrumSet` with metadata from CSV

---

## Exporters

### to_hdf5

Save `FoodSpectrumSet` to HDF5 format (fast, compact).

::: foodspec.io.exporters.to_hdf5
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.io import save_hdf5

# Save with HDF5
save_hdf5(
    fs,
    output_path="data/oils_library.h5"
)

# Check file size
import os
size_mb = os.path.getsize("data/oils_library.h5") / 1e6
print(f"HDF5 size: {size_mb:.1f} MB")
```

**Parameters:**
- `fs`: `FoodSpectrumSet` to save
- `output_path`: Path to `.h5` file

**See Also:** [`to_hdf5()`](#to_hdf5)

---

### to_tidy_csv

Export `FoodSpectrumSet` to tidy/long-form CSV.

::: foodspec.io.exporters.to_tidy_csv
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.io.exporters import to_tidy_csv

# Export to tidy format (one row per wavenumber per sample)
to_tidy_csv(fs, output_path="data/oils_tidy.csv")

# CSV structure:
# sample_id, oil_type, batch, wavenumber, intensity
# S001, Olive, A, 600.0, 0.523
# S001, Olive, A, 601.0, 0.531
# ...

# Load in pandas
import pandas as pd
df = pd.read_csv("data/oils_tidy.csv")
print(df.head())
```

**Parameters:**
- `fs`: `FoodSpectrumSet` to export
- `output_path`: Path to `.csv` file

**CSV Format:** Long-form with columns: `sample_id`, metadata columns, `wavenumber`, `intensity`

---

## Vendor Formats

### JCAMP-DX

::: foodspec.io.text_formats.read_jcamp
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.io import read_jcamp

# Read single JCAMP-DX file
wav, intensity, metadata_dict = read_jcamp("data/spectrum.jdx")

# Convert to FoodSpectrumSet
from foodspec.core import FoodSpectrumSet
import pandas as pd

fs = FoodSpectrumSet(
    x=intensity.reshape(1, -1),
    wavenumbers=wav,
    metadata=pd.DataFrame([metadata_dict]),
    modality="raman"
)
```

### Bruker OPUS

::: foodspec.io.vendor_formats.read_opus
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.io import read_opus

# Read Bruker OPUS file
wav, intensity = read_opus("data/sample.0")
print(f"Wavenumbers: {len(wav)}")
print(f"Intensity range: {intensity.min():.2f}-{intensity.max():.2f}")
```

---

## Cross-References

**Related Modules:**
- [Core](core.md) - `FoodSpectrumSet` data structure
- [Preprocessing](preprocessing.md) - Clean loaded data
- [Data Module](datasets.md) - Bundled example datasets

**Related Workflows:**
- [Oil Authentication](../workflows/authentication/oil_authentication.md) - Full loading → analysis pipeline
- [Data Formats Guide](../user-guide/data_formats_and_hdf5.md) - Detailed format documentation

