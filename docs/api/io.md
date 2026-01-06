# IO & Data Loading API

Functions for loading, saving, and converting spectral data across various formats.

The `foodspec.io` module handles data import/export with support for CSV, HDF5, JCAMP-DX, and vendor-specific formats (Bruker OPUS, SPC).

## Primary Functions

### load_folder

Load spectra from a directory of text files.

::: foodspec.io.loaders.load_folder
    options:
      show_source: false
      heading_level: 4

### read_spectra

Auto-detect format and read spectra.

::: foodspec.io.core.read_spectra
    options:
      show_source: false
      heading_level: 4

### detect_format

Identify file format by inspection.

::: foodspec.io.core.detect_format
    options:
      show_source: false
      heading_level: 4

## CSV & Text Formats

### load_csv_spectra

Load spectra from CSV files (wide or long format).

::: foodspec.io.csv_import.load_csv_spectra
    options:
      show_source: false
      heading_level: 4

### read_jcamp

Read JCAMP-DX spectroscopy files.

::: foodspec.io.text_formats.read_jcamp
    options:
      show_source: false
      heading_level: 4

## Vendor Formats

### read_opus

Read Bruker OPUS files (requires optional dependency).

::: foodspec.io.vendor_formats.read_opus
    options:
      show_source: false
      heading_level: 4

### read_spc

Read Thermo Galactic SPC files (requires optional dependency).

::: foodspec.io.vendor_formats.read_spc
    options:
      show_source: false
      heading_level: 4

## Export Functions

### to_hdf5

Save dataset to HDF5 format.

::: foodspec.io.exporters.to_hdf5
    options:
      show_source: false
      heading_level: 4

### to_tidy_csv

Export dataset to tidy (long-format) CSV.

::: foodspec.io.exporters.to_tidy_csv
    options:
      show_source: false
      heading_level: 4

## See Also

- **[Core Module](core.md)** - Data structures for loaded spectra
- **[Vendor Formats Guide](../user-guide/vendor_formats.md)** - Instrument file format details
- **[Examples](../examples_gallery.md)** - Loading examples

