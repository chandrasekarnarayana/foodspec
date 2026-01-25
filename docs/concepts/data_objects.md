# Data Objects

FoodSpec centers data objects around reproducible, metadata-rich spectroscopy.

## Core types
- `Spectrum`: single spectrum with axis metadata.
- `SpectraSet`: matrix of spectra with metadata table.
- `SpectralDataset`: protocol-aware dataset with provenance and IO helpers.

## Design notes
- Metadata is first-class and validated.
- Protocols and run metadata are stored alongside data objects when possible.

