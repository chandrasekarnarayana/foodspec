# Preprocessing API Reference

!!! info "Module Purpose"
    Transform raw spectra via baseline correction, smoothing, normalization, cropping using preprocessing pipelines.

---

## Quick Navigation

| Component | Purpose | Typical Use |
|-----------|---------|-------------|
| [`PreprocessPipeline`](#preprocesspipeline) | Chain preprocessing steps | Build reusable pipelines |
| [`BaselineStep`](#baseline-correction) | Baseline correction step | ALS, Rubberband, Polynomial |
| [`SmoothingStep`](#smoothing) | Savitzky-Golay smoothing | Noise reduction |
| [`NormalizationStep`](#normalization) | SNV, MSC, Vector normalization | Intensity standardization |
| [`AutoPreprocess`](#autopreprocess) | Automatic preprocessing | Quick preprocessing |

---

## Common Patterns

### Pattern 1: Pipeline-Based Preprocessing

```python
from foodspec.preprocess import (
    PreprocessPipeline,
    BaselineStep,
    SmoothingStep,
    NormalizationStep
)

# Build pipeline
pipeline = PreprocessPipeline(steps=[
    BaselineStep(method='als', lam=1e4, p=0.01),
    SmoothingStep(method='savgol', window_length=21, polyorder=3),
    NormalizationStep(method='snv')
])

# Apply to dataset
fs_processed = pipeline.transform(fs_raw)
print(f"Processed: {len(fs_processed)} spectra")

# Save pipeline for reproducibility
pipeline.save('pipelines/oil_preprocessing.json')
```

### Pattern 2: Auto-Preprocessing

```python
from foodspec.preprocess import AutoPreprocess

# Automatic preprocessing with sensible defaults
auto = AutoPreprocess(modality='raman')
result = auto.fit_transform(fs_raw)

fs_processed = result.transformed_dataset
print(f"Steps applied: {result.steps_applied}")
print(f"Parameters: {result.parameters}")
```

---

## PreprocessPipeline

Chain multiple preprocessing steps into a reproducible pipeline.

::: foodspec.preprocess.PreprocessPipeline
    options:
      show_source: false
      heading_level: 3

**Example:**

```python
from foodspec.preprocess import PreprocessPipeline, BaselineStep, NormalizationStep

# Create pipeline
pipeline = PreprocessPipeline(steps=[
    BaselineStep(method='als', lam=1e4),
    NormalizationStep(method='snv')
])

# Apply to data
fs_clean = pipeline.transform(fs_raw)

# Save/load
pipeline.save('my_pipeline.json')
loaded = PreprocessPipeline.load('my_pipeline.json')
```

**See Also:** [Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md)

---

## Baseline Correction

### BaselineStep

Baseline correction step for pipelines.

::: foodspec.preprocess.BaselineStep
    options:
      show_source: false
      heading_level: 4

**Supported Methods:**
- `'als'`: Asymmetric Least Squares (params: `lam`, `p`)
- `'rubberband'`: Convex hull (params: `n_points`)
- `'polynomial'`: Polynomial fitting (params: `degree`)

**Example:**

```python
from foodspec.preprocess import BaselineStep

# ALS baseline
step_als = BaselineStep(method='als', lam=1e4, p=0.01)

# Rubberband baseline
step_rubber = BaselineStep(method='rubberband', n_points=100)

# Polynomial baseline
step_poly = BaselineStep(method='polynomial', degree=3)
```

**See Also:** [Baseline Correction Theory](../methods/preprocessing/baseline_correction.md)

---

## Smoothing

### SmoothingStep

Savitzky-Golay smoothing and derivatives.

::: foodspec.preprocess.SmoothingStep
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.preprocess import SmoothingStep

# Smoothing
step_smooth = SmoothingStep(
    method='savgol',
    window_length=21,
    polyorder=3,
    deriv=0  # 0=smooth, 1=1st deriv, 2=2nd deriv
)

# 1st derivative
step_deriv1 = SmoothingStep(
    method='savgol',
    window_length=21,
    polyorder=3,
    deriv=1
)
```

---

## Normalization

### NormalizationStep

Normalization step for intensity standardization.

::: foodspec.preprocess.NormalizationStep
    options:
      show_source: false
      heading_level: 4

**Supported Methods:**
- `'snv'`: Standard Normal Variate
- `'msc'`: Multiplicative Scatter Correction
- `'l1'`, `'l2'`, `'max'`: Vector normalization

**Example:**

```python
from foodspec.preprocess import NormalizationStep

# SNV normalization
step_snv = NormalizationStep(method='snv')

# MSC normalization
step_msc = NormalizationStep(method='msc')

# L2 normalization
step_l2 = NormalizationStep(method='l2')
```

---

## AutoPreprocess

Automatic preprocessing with sensible defaults for each modality.

::: foodspec.preprocess.AutoPreprocess
    options:
      show_source: false
      heading_level: 3

**Example:**

```python
from foodspec.preprocess import AutoPreprocess

# Auto-preprocess Raman spectra
auto_raman = AutoPreprocess(modality='raman')
result = auto_raman.fit_transform(fs_raman)

print(f"Steps: {result.steps_applied}")
print(f"Parameters: {result.parameters}")

# Auto-preprocess FTIR spectra
auto_ftir = AutoPreprocess(modality='ftir')
result_ftir = auto_ftir.fit_transform(fs_ftir)
```

---

## Cross-References

**Related Modules:**
- [Core](core.md) - `FoodSpectrumSet` data structure
- [IO](io.md) - Load raw data
- [Features](features.md) - Extract features from preprocessed data

**Related Workflows:**
- [Oil Authentication](../workflows/authentication/oil_authentication.md) - Full preprocessing pipeline
- [Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md) - Parameter selection
