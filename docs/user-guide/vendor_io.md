# Vendor IO and Harmonization

**Purpose:** Load spectra from vendor instruments (OPUS, WiRE, ENVI) and harmonize data across instruments.

**Audience:** Lab scientists with instrument-specific data; data managers integrating multi-instrument datasets.

**Time:** 10–15 minutes to set up loader; ongoing monitoring for harmonization.

**Prerequisites:** Understand data format expectations; familiarity with CSV or HDF5 basics.

---

## Supported Formats (Current)

FoodSpec supports several input formats. Choose based on your data:

| Format | Use When | Example |
|--------|----------|---------|
| CSV (wide) | Quick start, small datasets | `load_csv("data.csv", wavenumber_col="wn")` |
| HDF5 | Multi-instrument projects, HSI | `load_hdf5("data.h5", dataset_path="/spectra")` |
| OPUS (ASCII) | Bruker FTIR instruments | `load_opus("data_export.txt")` |
| WiRE (ASCII) | Renishaw Raman systems | `load_wire("export.txt")` |
| ENVI | Hyperspectral imaging | `load_envi("cube.img", "cube.hdr")` |

**Example: Load OPUS file and harmonize**

```python
from foodspec.io import load_opus
from foodspec.preprocess import calibrate_wavenumbers
import numpy as np

# Load and export as CSV
ds_opus = load_opus("sample.txt")
print(f"Loaded {len(ds_opus)} spectra from OPUS")

# If wavenumber drift detected, harmonize
ds_harmonized = calibrate_wavenumbers(
    ds_opus,
    reference_wavenumbers=np.arange(400, 3200, 1),
    method="piecewise_direct_standardization"
)

# Save as HDF5 for reproducibility
ds_harmonized.to_hdf5("harmonized_data.h5")
```

## Limitations
- Binary vendor formats are not fully parsed; export to ASCII/CSV/HDF5 for best results.
- If a file looks like a vendor format but fails to parse, error messages will suggest exporting paths.

## Harmonization
- Use calibration curves per instrument to correct wavenumber drift; store in instrument metadata.
- Normalize intensities using laser power metadata when available.
- Diagnostics: residual variation and pre/post plots.

## Extending Vendor IO
- Add loaders via plugins (`foodspec.plugins` entry point). See `examples/plugins/plugin_example_vendor`.

## Next Steps

- **Troubleshooting:** [Vendor format issues](../help/troubleshooting.md#vendor-io-errors)
- **Data format details:** [CSV/HDF5 schema](data_formats_and_hdf5.md)
- **Plugins:** [Register custom loaders](registry_and_plugins.md)
- **Multi-instrument workflows:** [Harmonization guide](../workflows/harmonization_automated_calibration.md)
