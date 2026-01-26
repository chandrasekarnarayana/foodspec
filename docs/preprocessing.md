# FoodSpec Preprocessing Engine

Comprehensive spectroscopy preprocessing for Raman, FTIR, and NIR data.

## Overview

The FoodSpec preprocessing engine provides a robust, extensible system for spectroscopic data preprocessing with:

- **Raman-specific**: Cosmic ray removal (despiking), fluorescence baseline removal
- **FTIR/IR-specific**: MSC/EMSC scatter correction, atmospheric absorption correction
- **Shared operations**: Savitzky-Golay smoothing, derivatives, normalization (SNV/area/vector), baseline correction (ALS/polynomial/SNIP/rubberband), alignment, interpolation

### Key Features

✓ **YAML Recipe System**: Reusable preprocessing workflows defined in YAML  
✓ **Preset Library**: Pre-configured recipes for common tasks (`raman`, `ftir`, `oil_auth`)  
✓ **Composable Pipelines**: Chain operators in any order  
✓ **Caching & Provenance**: Hash-based caching with full metadata tracking  
✓ **QC Visualization**: Automatic generation of raw vs processed overlays  
✓ **Deterministic**: Reproducible results with seed control  
✓ **Protocol Integration**: Seamless integration with FoodSpec protocol system  

---

## Quick Start

### Python API

```python
from foodspec.preprocess import load_recipe, PreprocessPipeline
from foodspec.data_objects.spectra_set import FoodSpectrumSet
import numpy as np

# Load data
ds = FoodSpectrumSet(
    x=np.random.rand(100, 512),
    wavenumbers=np.linspace(500, 3000, 512),
    modality="raman",
)

# Load preset recipe
pipeline = load_recipe(preset="raman")

# Run preprocessing
result, metrics = pipeline.transform(ds)

print(f"Processed shape: {result.x.shape}")
print(f"Metrics: {metrics}")
```

### YAML Protocol

```yaml
protocol:
  name: "Oil_Authentication_v1"
  modality: raman
  
  steps:
    - type: preprocess
      preset: oil_auth  # Use preset from library
      
    - type: feature_extraction
      method: pca
      n_components: 50
```

---

## Preprocessing Operators

### Raman-Specific

#### Despiking (Cosmic Ray Removal)

Remove cosmic ray spikes using median filtering.

```yaml
- op: despike
  window: 5
  threshold: 5.0
```

**Parameters:**
- `window` (int): Median filter window size (default 5)
- `threshold` (float): Z-score threshold for spike detection (default 5.0)

**Python:**
```python
from foodspec.preprocess.spectroscopy_operators import DespikeOperator

op = DespikeOperator(window=5, threshold=5.0)
result = op.transform(ds)
```

#### Fluorescence Removal

Remove Raman fluorescence background.

```yaml
- op: fluorescence_removal
  method: poly
  poly_order: 2
```

**Parameters:**
- `method` ({'poly', 'als'}): Background estimation method
- `poly_order` (int): Polynomial order if method='poly' (default 2)

---

### FTIR-Specific

#### EMSC (Extended Multiplicative Scatter Correction)

Correct for multiplicative scatter and additive baseline effects.

```yaml
- op: emsc
  order: 2
```

**Parameters:**
- `reference` (array | None): Reference spectrum (default: mean spectrum)
- `order` (int): Polynomial order for correction (default 2)

#### MSC (Multiplicative Scatter Correction)

Simpler scatter correction without polynomial baseline.

```yaml
- op: msc
```

#### Atmospheric Correction

Remove common atmospheric absorption lines (CO₂, H₂O).

```yaml
- op: atmospheric_correction
  co2_window: 50
  water_window: 100
```

**Parameters:**
- `co2_window` (int): Window around CO₂ line (2350 cm⁻¹)
- `water_window` (int): Window around water lines (1600-1800 cm⁻¹)

---

### Shared Operators

#### Baseline Correction

Remove baseline using various methods.

```yaml
- op: baseline
  method: als
  lam: 1.0e5
  p: 0.01
```

**Methods:**
- `als`: Asymmetric least squares (recommended)
- `airpls`: Adaptive iteratively reweighted penalized least squares
- `snip`: Statistics-sensitive non-linear iterative peak-clipping
- `poly`: Polynomial fitting
- `rubberband`: Convex hull baseline

**ALS Parameters:**
- `lam` (float): Smoothness parameter (1e4-1e6, default 1e5)
- `p` (float): Asymmetry parameter (0.001-0.01, default 0.01)

#### Smoothing

Apply smoothing filters.

```yaml
- op: smoothing
  method: savgol
  window_length: 7
  polyorder: 3
```

**Methods:**
- `savgol`: Savitzky-Golay filter (preserves peaks)
- `gaussian`: Gaussian filter
- `moving_average`: Simple moving average

**Savitzky-Golay Parameters:**
- `window_length` (int): Window size (must be odd, default 7)
- `polyorder` (int): Polynomial order (< window_length, default 3)

#### Normalization

Normalize spectra to standard scale.

```yaml
- op: normalization
  method: snv
```

**Methods:**
- `snv`: Standard Normal Variate (mean=0, std=1)
- `vector`: Unit vector normalization (L2 norm=1)
- `area`: Area normalization (sum=1)
- `max`: Max normalization (max=1)
- `msc`: Multiplicative scatter correction

#### Derivatives

Compute spectral derivatives.

```yaml
- op: derivative
  order: 1
  window_length: 9
  polyorder: 2
```

**Parameters:**
- `order` (int): Derivative order (1 or 2)
- `window_length` (int): Savitzky-Golay window (default 7)
- `polyorder` (int): Polynomial order (default 2)

#### Interpolation

Align to reference wavenumber grid.

```yaml
- op: interpolation
  method: linear
  # target_grid defined in code
```

---

## Preset Library

### Available Presets

| Preset | Modality | Description |
|--------|----------|-------------|
| `default` | All | Conservative baseline + smoothing + SNV |
| `raman` | Raman | Despike + fluorescence + baseline + smoothing + SNV |
| `ftir` | FTIR | Atmospheric correction + MSC + baseline + smoothing + area norm |
| `oil_auth` | Raman | Optimized for edible oil authentication |
| `chips_matrix` | FTIR | Optimized for complex food matrices |

### Preset: `default`

```yaml
modality: unknown
description: "Safe baseline, smoothing, and normalization"
steps:
  - op: baseline
    method: als
    lam: 1.0e5
    p: 0.01
  - op: smoothing
    method: savgol
    window_length: 7
    polyorder: 3
  - op: normalization
    method: snv
```

### Preset: `raman`

```yaml
modality: raman
description: "Raman preset: cosmic ray removal, fluorescence removal, smoothing, SNV"
steps:
  - op: despike
    window: 5
    threshold: 5.0
  - op: fluorescence_removal
    method: poly
    poly_order: 2
  - op: smoothing
    method: savgol
    window_length: 9
    polyorder: 3
  - op: baseline
    method: als
    lam: 1.0e5
    p: 0.01
  - op: normalization
    method: snv
```

### Preset: `ftir`

```yaml
modality: ftir
description: "FTIR preset: atmospheric correction, MSC, baseline, smoothing"
steps:
  - op: atmospheric_correction
    co2_window: 50
    water_window: 100
  - op: msc
  - op: baseline
    method: rubberband
  - op: smoothing
    method: savgol
    window_length: 7
    polyorder: 3
  - op: normalization
    method: area
```

---

## Recipe System

### Loading Recipes

#### From Preset

```python
from foodspec.preprocess import load_recipe

pipeline = load_recipe(preset="raman")
```

#### From Protocol Config

```python
protocol_config = {
    "preprocess": {
        "modality": "raman",
        "steps": [
            {"op": "despike", "window": 5},
            {"op": "baseline", "method": "als"},
        ]
    }
}

pipeline = load_recipe(protocol_config=protocol_config)
```

#### With CLI Overrides

```python
cli_overrides = {
    "override_steps": [
        {"op": "baseline", "lam": 1e6}  # Override baseline parameter
    ]
}

pipeline = load_recipe(preset="default", cli_overrides=cli_overrides)
```

### Building Custom Recipes

```python
from foodspec.preprocess import PreprocessPipeline
from foodspec.engine.preprocessing.engine import BaselineStep, SmoothingStep, NormalizationStep

# Build custom pipeline
pipeline = PreprocessPipeline()
pipeline.add(BaselineStep(method="als", lam=1e5))
pipeline.add(SmoothingStep(method="savgol", window_length=9, polyorder=3))
pipeline.add(NormalizationStep(method="snv"))

# Run
result, metrics = pipeline.transform(ds)
```

---

## Caching & Provenance

### Automatic Caching

```python
from foodspec.preprocess.cache import PreprocessCache, compute_data_hash, compute_recipe_hash, compute_cache_key

# Initialize cache
cache = PreprocessCache(cache_dir="./cache")

# Compute cache key
data_hash = compute_data_hash(ds.x, ds.wavenumbers)
recipe_hash = compute_recipe_hash(recipe_dict)
cache_key = compute_cache_key(data_hash, recipe_hash, seed=42)

# Check cache
cached = cache.get(cache_key)
if cached:
    X_processed = cached["X"]
else:
    # Run preprocessing
    result, _ = pipeline.transform(ds)
    cache.put(cache_key, result.x, result.wavenumbers)
```

### Provenance Tracking

```python
from foodspec.preprocess.cache import PreprocessManifest

# Create manifest
manifest = PreprocessManifest(
    run_id="exp_20250126_001",
    recipe=recipe_dict,
    cache_key=cache_key,
    seed=42,
)

# Record operator execution
manifest.record_operator("despike", time_ms=12.3, spikes_removed=5)
manifest.record_operator("baseline", time_ms=45.6)

# Add warnings
manifest.add_warning("Sample 10 had NaN values after baseline correction")

# Finalize
manifest.finalize(
    n_samples_input=100,
    n_samples_output=98,
    n_features=512,
    rejected_spectra=2,
    rejection_reasons=["nan_values", "cosmic_ray"],
)

# Save
manifest.save("outdir/data/preprocess_manifest.json")
```

### Manifest JSON Structure

```json
{
  "run_id": "exp_20250126_001",
  "recipe": {
    "modality": "raman",
    "steps": [...]
  },
  "cache_key": "abc123...",
  "seed": 42,
  "foodspec_version": "2.0.0",
  "timestamps": {
    "start": "2025-01-26T15:06:30",
    "end": "2025-01-26T15:06:35"
  },
  "statistics": {
    "n_samples_input": 100,
    "n_samples_output": 98,
    "n_features": 512,
    "rejected_spectra": 2,
    "rejection_reasons": ["nan_values"],
    "duration_seconds": 5.2
  },
  "operators_applied": [
    {"op": "despike", "time_ms": 12.3, "spikes_removed": 5},
    {"op": "baseline", "time_ms": 45.6}
  ],
  "warnings": ["Sample 10 had NaN values"]
}
```

---

## QC Visualization

### Generate QC Plots

```python
from foodspec.preprocess.qc import generate_qc_report

# Generate all QC plots
generate_qc_report(
    X_raw=ds_raw.x,
    X_processed=ds_processed.x,
    wavenumbers=ds_raw.wavenumbers,
    baselines=baseline_estimates,  # Optional
    outlier_mask=outlier_mask,      # Optional
    output_dir="outdir/figures",
)
```

### Generated Plots

1. **raw_vs_processed_overlay.png**: Overlay of 5 sampled spectra (raw vs processed)
2. **baseline_estimate_overlay.png**: Baseline estimates for sampled spectra
3. **outlier_detection_summary.png**: Histograms of spectral norms and distances

---

## Protocol Integration

### Enhanced PreprocessStep

The preprocessing engine integrates with FoodSpec's protocol system via `PreprocessStep`.

```yaml
protocol:
  name: "Oil_Authentication"
  steps:
    - type: preprocess
      preset: oil_auth
      override_steps:
        - op: baseline
          lam: 5.0e5  # Override parameter
```

```python
# In protocol runner
from foodspec.protocol.steps import PreprocessStep

step = PreprocessStep(cfg={
    "preset": "raman",
    "override_steps": [{"op": "baseline", "lam": 1e6}]
})

ctx = {"data": ds}
step.run(ctx)
# ctx["data"] now contains preprocessed data
```

---

## Data Loading

### Load CSV (Wide Format)

```python
from foodspec.preprocess.data import load_csv

# Wide format: columns = wavenumbers
data = load_csv("data.csv", format="wide", modality="raman")

# data.X: (n_samples, n_features)
# data.wavenumbers: (n_features,)
# data.metadata: DataFrame with sample-level info
```

### Load CSV (Long Format)

```python
# Long format: columns = sample_id, wavenumber, intensity
data = load_csv("long_data.csv", format="long")
```

### Auto-detect Format

```python
# Automatically detect format
data = load_csv("data.csv", format="auto")
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Issue**: `ImportError: cannot import name 'DespikeOperator'`

**Solution**: Ensure FoodSpec is installed with preprocessing extras:
```bash
pip install -e ".[preprocess]"
```

#### 2. Preset Not Found

**Issue**: `ValueError: Unknown preset: custom_preset`

**Solution**: Use one of the built-in presets (`default`, `raman`, `ftir`, `oil_auth`, `chips_matrix`) or add custom preset to `src/foodspec/preprocess/presets/custom/`.

#### 3. Inconsistent Results

**Issue**: Same data produces different results

**Solution**: Set random seed for deterministic behavior:
```python
import numpy as np
np.random.seed(42)

pipeline = load_recipe(preset="raman")
result, _ = pipeline.transform(ds)
```

#### 4. NaN Values After Preprocessing

**Issue**: `result.x` contains NaN values

**Solution**: Check for invalid input data or extreme parameter values. Validate input:
```python
assert np.all(np.isfinite(ds.x)), "Input contains NaN/Inf"
```

#### 5. Slow Preprocessing

**Issue**: Preprocessing takes too long

**Solution**: Enable caching:
```python
cache = PreprocessCache(cache_dir="./cache")
cache_key = compute_cache_key(data_hash, recipe_hash)
cached = cache.get(cache_key)
if cached is None:
    # Run preprocessing only if not cached
    result, _ = pipeline.transform(ds)
    cache.put(cache_key, result.x)
```

---

## API Reference

### Core Classes

- `PreprocessPipeline`: Orchestrate sequence of preprocessing steps
- `Step`: Base class for all operators
- `SpectraData`: Standard internal representation

### Operators

**Raman**: `DespikeOperator`, `FluorescenceRemovalOperator`  
**FTIR**: `EMSCOperator`, `MSCOperator`, `AtmosphericCorrectionOperator`  
**Shared**: `BaselineStep`, `SmoothingStep`, `NormalizationStep`, `DerivativeStep`, `InterpolationOperator`

### Recipe Loading

- `load_preset_yaml(name)`: Load YAML preset
- `build_pipeline_from_recipe(dict)`: Build pipeline from recipe dict
- `load_recipe(preset, protocol_config, cli_overrides)`: Load and merge recipes
- `list_operators()`: List all available operators

### Caching

- `PreprocessCache`: File-based cache
- `PreprocessManifest`: Provenance tracking
- `compute_data_hash`, `compute_recipe_hash`, `compute_cache_key`: Hashing utilities

### QC

- `plot_raw_vs_processed`: Raw vs processed overlay
- `plot_baseline_overlay`: Baseline estimates
- `plot_outlier_summary`: Outlier detection
- `generate_qc_report`: Generate all plots

---

## Testing

### Run Tests

```bash
# All preprocessing tests
pytest tests/preprocess/ -v

# Specific test file
pytest tests/preprocess/test_integration.py -v

# With coverage
pytest tests/preprocess/ --cov=src/foodspec/preprocess
```

### Test Coverage

- Data loading (wide/long CSV)
- All operators (unit tests)
- Recipe loading and merging
- Caching correctness
- Deterministic behavior
- Integration with protocol system

---

## Examples

### Example 1: Raman Oil Authentication

```python
from foodspec.preprocess import load_recipe
from foodspec.data_objects.spectra_set import FoodSpectrumSet
import pandas as pd
import numpy as np

# Load data
X = np.load("oils_raman.npy")
wavenumbers = np.load("wavenumbers.npy")
metadata = pd.read_csv("oils_metadata.csv")

ds = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=metadata, modality="raman")

# Load preset
pipeline = load_recipe(preset="oil_auth")

# Run preprocessing
result, metrics = pipeline.transform(ds)

# Save
np.save("oils_preprocessed.npy", result.x)
print(f"Preprocessing metrics: {metrics}")
```

### Example 2: FTIR Custom Pipeline

```python
from foodspec.preprocess import PreprocessPipeline
from foodspec.engine.preprocessing.engine import BaselineStep, SmoothingStep, NormalizationStep
from foodspec.preprocess.spectroscopy_operators import MSCOperator

# Build custom pipeline
pipeline = PreprocessPipeline()
pipeline.add(MSCOperator())
pipeline.add(BaselineStep(method="als", lam=5e5, p=0.001))
pipeline.add(SmoothingStep(method="savgol", window_length=9, polyorder=3))
pipeline.add(NormalizationStep(method="area"))

# Load data
ds = ...  # FoodSpectrumSet

# Run
result, metrics = pipeline.transform(ds)
```

### Example 3: Protocol with Preprocessing

```yaml
protocol:
  name: "Oil_Authentication_Full_Pipeline"
  modality: raman
  
  steps:
    - type: preprocess
      preset: oil_auth
      override_steps:
        - op: baseline
          lam: 5.0e5
    
    - type: feature_extraction
      method: pca
      n_components: 50
    
    - type: classification
      model: lightgbm
      validation: lobo
```

---

## Contributing

To add a new operator:

1. Create operator class in `src/foodspec/preprocess/spectroscopy_operators.py` or `src/foodspec/engine/preprocessing/engine.py`
2. Register in `src/foodspec/preprocess/loaders.py` (`_OPERATOR_REGISTRY`)
3. Add tests in `tests/preprocess/test_operators.py`
4. Update documentation

---

## References

- **ALS Baseline**: Eilers, P.H., & Boelens, H.F. (2005). Baseline correction with asymmetric least squares smoothing. *Leiden University Medical Centre Report*, 1(1).
- **EMSC**: Martens, H., & Stark, E. (1991). Extended multiplicative signal correction and spectral interference subtraction. *Journal of Pharmaceutical and Biomedical Analysis*, 9(8), 625-635.
- **Savitzky-Golay**: Savitzky, A., & Golay, M.J. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627-1639.

---

## License

FoodSpec preprocessing engine is part of FoodSpec, licensed under the MIT License.
