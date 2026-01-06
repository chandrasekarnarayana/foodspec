# Features API

Feature extraction and peak analysis functions for spectral data.

The `foodspec.features` module provides tools for extracting meaningful features from spectra, including peak detection, band integration, and ratio-based quality metrics.

## Peak Detection

### detect_peaks

Detect spectral peaks with prominence and width filtering.

::: foodspec.features.peaks.detect_peaks
    options:
      show_source: false
      heading_level: 4

## Band Integration

### compute_band_features

Compute features from spectral bands (integral, mean, max, slope).

::: foodspec.features.bands.compute_band_features
    options:
      show_source: false
      heading_level: 4

### integrate_bands

Legacy wrapper for band integration.

::: foodspec.features.bands.integrate_bands
    options:
      show_source: false
      heading_level: 4

## Similarity Metrics

### cosine_similarity_matrix

Compute pairwise cosine similarity between spectra.

::: foodspec.features.fingerprint.cosine_similarity_matrix
    options:
      show_source: false
      heading_level: 4

### correlation_similarity_matrix

Compute pairwise Pearson correlation between spectra.

::: foodspec.features.fingerprint.correlation_similarity_matrix
    options:
      show_source: false
      heading_level: 4

## Ratio Quality (RQ) Engine

### RatioQualityEngine

Automated peak ratio-based quality assessment workflow.

::: foodspec.features.rq.engine.RatioQualityEngine
    options:
      show_source: false
      heading_level: 4

### RQConfig

Configuration for ratio quality analysis.

::: foodspec.features.rq.types.RQConfig
    options:
      show_source: false
      heading_level: 4

### PeakDefinition

Define spectral peaks for ratio analysis.

::: foodspec.features.rq.types.PeakDefinition
    options:
      show_source: false
      heading_level: 4

### RatioDefinition

Define peak ratios for quality metrics.

::: foodspec.features.rq.types.RatioDefinition
    options:
      show_source: false
      heading_level: 4

## Library Search

### similarity_search

Find nearest neighbors in a spectral library.

::: foodspec.features.library.similarity_search
    options:
      show_source: false
      heading_level: 4

## See Also

- **[Feature Extraction Methods](../methods/preprocessing/feature_extraction.md)** - Detailed methodology
- **[RQ Engine Theory](../theory/rq_engine_detailed.md)** - Ratio quality concepts
- **[Examples](../examples_gallery.md)** - Feature extraction workflows
