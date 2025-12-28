# Vendor IO and Harmonization Notes

## Supported formats (current)
- CSV/TXT (wide-format): preferred for reliability.
- HDF5 (FoodSpec NeXus-like schema): recommended for full metadata/provenance.
- Vendor stubs:
  - OPUS (ASCII export): use `load_opus`.
  - Renishaw WiRE (ASCII export): use `load_wire`.
  - ENVI header/data (ASCII): use `load_envi`.

## Limitations
- Binary vendor formats are not fully parsed; export to ASCII/CSV/HDF5 for best results.
- If a file looks like a vendor format but fails to parse, error messages will suggest exporting paths.

## Harmonization
- Use calibration curves per instrument to correct wavenumber drift; store in instrument metadata.
- Normalize intensities using laser power metadata when available.
- Diagnostics: residual variation and pre/post plots.

## Extending vendor IO
- Add loaders via plugins (`foodspec.plugins` entry point). See `examples/plugins/plugin_example_vendor`.
