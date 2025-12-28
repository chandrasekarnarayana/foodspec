# Features API Reference

!!! info "Module Purpose"
    Feature extraction from spectra: peak detection, band integration, peak ratios, and the Ratio-Quality (RQ) engine.

---

## Quick Navigation

| Function/Class | Purpose | Typical Use |
|----------------|---------|-------------|
| [`detect_peaks()`](#peak-detection) | Find spectral peaks | Peak-based analysis |
| [`integrate_bands()`](#band-integration) | Integrate spectral regions | Quantitative band analysis |
| [`compute_ratios()`](#peak-ratios) | Calculate peak/band ratios | Ratio-based features |
| [`RatioQualityEngine`](#rq-engine) | RQ workflow | Automated ratio analysis |
| [`PeakFeatureExtractor`](#peakfeatureextractor) | Extract peak features | Height, width, area |

---

## Common Patterns

### Pattern 1: Peak Detection

```python
from foodspec.features import detect_peaks

# Detect peaks
peaks = detect_peaks(
    spectrum,
    wavenumbers,
    height=0.1,
    prominence=0.05,
    distance=10
)

print(f"Found {len(peaks['positions'])} peaks")
print(f"Peak wavenumbers: {peaks['wavenumbers']}")
```

### Pattern 2: Band Integration

```python
from foodspec.features import integrate_bands

# Define bands
bands = {
    'C-H stretch': (2800, 3000),
    'C=O stretch': (1700, 1750),
}

# Integrate
band_areas = integrate_bands(fs, bands, method='trapezoid')
print(f"C-H area: {band_areas['C-H stretch'].mean():.2f}")
```

### Pattern 3: Ratio Features

```python
from foodspec.features import compute_ratios

# Define peaks
peaks = {'peak_A': 1650, 'peak_B': 1450}

# Compute ratios
ratios = compute_ratios(fs, peaks, method='height')
ratio_AB = ratios['peak_A/peak_B']
print(f"A/B ratio: {ratio_AB.mean():.3f}")
```

---

## Peak Detection

### detect_peaks

::: foodspec.features.detect_peaks
    options:
      show_source: false

### PeakFeatureExtractor

::: foodspec.features.PeakFeatureExtractor
    options:
      show_source: false

---

## Band Integration

### integrate_bands

::: foodspec.features.integrate_bands
    options:
      show_source: false

---

## Peak Ratios

### compute_ratios

::: foodspec.features.compute_ratios
    options:
      show_source: false

### RatioFeatureGenerator

::: foodspec.features.RatioFeatureGenerator
    options:
      show_source: false

---

## RQ Engine

### RatioQualityEngine

::: foodspec.features.RatioQualityEngine
    options:
      show_source: false

### RatioQualityResult

::: foodspec.features.RatioQualityResult
    options:
      show_source: false

### PeakDefinition

::: foodspec.features.PeakDefinition
    options:
      show_source: false

---

## Feature Specification

### FeatureSpec

::: foodspec.features.FeatureSpec
    options:
      show_source: false

### FeatureEngine

::: foodspec.features.FeatureEngine
    options:
      show_source: false

---

## Cross-References

**Related Modules:**
- [Core](core.md) - FoodSpectrumSet data structure
- [Preprocessing](preprocessing.md) - Preprocess before extraction
- [Chemometrics](chemometrics.md) - Use features in models

**Related Workflows:**
- [Ratio-Quality Demo](../workflows/domain_templates.md) - RQ engine example
- [Heating Quality](../workflows/quality-monitoring/heating_quality_monitoring.md) - Ratio-based monitoring
