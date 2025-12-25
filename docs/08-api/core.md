# Core API Reference

The core module provides the fundamental building blocks for FoodSpec workflows.

## FoodSpectrumSet

Primary data container for spectral data (Raman, FTIR, NIR).

::: foodspec.core.dataset.FoodSpectrumSet
    options:
      show_source: false
      heading_level: 3

## FoodSpec High-Level API

High-level interface for analysis workflows.

::: foodspec.core.api.FoodSpec
    options:
      show_source: false
      heading_level: 3

## HyperSpectralCube

Hyperspectral imaging data structure.

::: foodspec.core.hyperspectral.HyperSpectralCube
    options:
      show_source: false
      heading_level: 3

## MultiModalDataset

Combined spectral modalities (e.g., Raman + NIR fusion).

::: foodspec.core.multimodal.MultiModalDataset
    options:
      show_source: false
      heading_level: 3

## OutputBundle

Reproducible output management.

::: foodspec.core.output_bundle.OutputBundle
    options:
      show_source: false
      heading_level: 3

## RunRecord

Experiment tracking and reproducibility.

::: foodspec.core.run_record.RunRecord
    options:
      show_source: false
      heading_level: 3

---

## Usage Examples

### Creating a FoodSpectrumSet

```python
from foodspec.core import FoodSpectrumSet
import numpy as np
import pandas as pd

# From arrays
X = np.random.randn(100, 500)
wavenumbers = np.linspace(1000, 3000, 500)
metadata = pd.DataFrame({'sample_id': range(100), 'label': ['A']*50 + ['B']*50})

dataset = FoodSpectrumSet(
    x=X, 
    wavenumbers=wavenumbers, 
    metadata=metadata,
    modality='raman'
)
```

### Using the High-Level API

```python
from foodspec.core import FoodSpec

fs = FoodSpec(dataset)
fs.preprocess(baseline='rubberband', normalize='vector')
fs.pca(n_components=3)
fs.plot_pca_scores(color_by='label')
```
