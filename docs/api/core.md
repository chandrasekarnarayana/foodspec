# Core API

Core data structures and workflows for spectral analysis.

The `foodspec.core` module provides foundational classes for working with spectral data, including dataset containers, single spectrum operations, and result packaging.

## Main Classes

### FoodSpectrumSet

Primary container for spectral datasets with aligned metadata.

::: foodspec.core.dataset.FoodSpectrumSet
    options:
      show_source: false
      heading_level: 4

### Spectrum

Single spectrum data model with validation.

::: foodspec.core.spectrum.Spectrum
    options:
      show_source: false
      heading_level: 4

### OutputBundle

Structured output packaging for analysis results.

::: foodspec.core.output_bundle.OutputBundle
    options:
      show_source: false
      heading_level: 4

### RunRecord

Provenance tracking for reproducible analyses.

::: foodspec.core.run_record.RunRecord
    options:
      show_source: false
      heading_level: 4

## Advanced Data Structures

### SpectralDataset

Extended dataset with preprocessing pipeline integration.

::: foodspec.core.spectral_dataset.SpectralDataset
    options:
      show_source: false
      heading_level: 4

### HyperspectralDataset

Hyperspectral imaging (3D spatial-spectral) data container.

::: foodspec.core.spectral_dataset.HyperspectralDataset
    options:
      show_source: false
      heading_level: 4

## Helper Functions

### Conversion Utilities

::: foodspec.core.dataset.to_sklearn
    options:
      show_source: false
      heading_level: 4

::: foodspec.core.dataset.from_sklearn
    options:
      show_source: false
      heading_level: 4

## See Also

- **[IO Module](io.md)** - Loading and saving spectral data
- **[Preprocessing](../methods/preprocessing/baseline_correction.md)** - Data cleaning methods
- **[Examples](../examples_gallery.md)** - Practical usage examples

