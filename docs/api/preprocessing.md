# Preprocessing API

Spectral preprocessing functions for baseline correction, normalization, and noise reduction.

The `foodspec.preprocess` module provides tools for cleaning and normalizing spectral data before analysis.

## Baseline Correction

### baseline_als

Asymmetric Least Squares baseline correction.

::: foodspec.core.spectral_dataset.baseline_als
    options:
      show_source: false
      heading_level: 4

### baseline_polynomial

Polynomial fitting baseline correction.

::: foodspec.core.spectral_dataset.baseline_polynomial
    options:
      show_source: false
      heading_level: 4

### baseline_rubberband

Rubberband baseline correction.

::: foodspec.core.spectral_dataset.baseline_rubberband
    options:
      show_source: false
      heading_level: 4

## Normalization

### VectorNormalizer

L2 vector normalization.

::: foodspec.preprocess.normalization.VectorNormalizer
    options:
      show_source: false
      heading_level: 4

### SNVNormalizer

Standard Normal Variate normalization.

::: foodspec.preprocess.normalization.SNVNormalizer
    options:
      show_source: false
      heading_level: 4

### AreaNormalizer

Area-under-curve normalization.

::: foodspec.preprocess.normalization.AreaNormalizer
    options:
      show_source: false
      heading_level: 4

## Smoothing

### SavitzkyGolaySmoother

Savitzky-Golay filter for smoothing and derivatives.

::: foodspec.preprocess.smoothing.SavitzkyGolaySmoother
    options:
      show_source: false
      heading_level: 4

### MovingAverageSmoother

Simple moving average smoothing.

::: foodspec.preprocess.smoothing.MovingAverageSmoother
    options:
      show_source: false
      heading_level: 4

## Noise & Artifact Removal

### correct_cosmic_rays

Remove cosmic ray spikes from Raman spectra.

::: foodspec.preprocess.spikes.correct_cosmic_rays
    options:
      show_source: false
      heading_level: 4

### CosmicRayRemover

Advanced cosmic ray removal for Raman spectroscopy.

::: foodspec.preprocess.raman.CosmicRayRemover
    options:
      show_source: false
      heading_level: 4

## Spectral Cropping

### RangeCropper

Crop spectra to wavenumber ranges.

::: foodspec.preprocess.cropping.RangeCropper
    options:
      show_source: false
      heading_level: 4

## See Also

- **[Preprocessing Methods](../methods/preprocessing/baseline_correction.md)** - Detailed methodology
- **[Examples](../examples_gallery.md)** - Preprocessing workflows
- **[Core Module](core.md)** - Dataset structures
