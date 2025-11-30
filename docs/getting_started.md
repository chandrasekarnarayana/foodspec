# Getting started

This page walks you through installing foodspec and running your first analysis.

## Installation

```bash
pip install foodspec

# For optional deep-learning support (1D CNN classifier)
pip install 'foodspec[deep]'
```

## First steps with FoodSpectrumSet

The core object in foodspec is `FoodSpectrumSet`, which bundles:

- a 2D array of intensities (`x`),
- a shared wavenumber axis (`wavenumbers`),
- a metadata table (`metadata`),
- and a modality tag (`modality`).

### Minimal example (Python API)

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.validation import validate_spectrum_set
from foodspec.chemometrics.pca import run_pca

# 1. Create or load a dataset (replace with real data or load_example_oils)
wn = np.linspace(1600, 1700, 50)
X = np.random.rand(5, wn.size)
meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(5)], "oil_type": ["A", "A", "B", "B", "B"]})
fs = FoodSpectrumSet(x=X, wavenumbers=wn, metadata=meta, modality="raman")
validate_spectrum_set(fs)

# 2. Build and apply a preprocessing pipeline
pipe = Pipeline(
    [
        ("als", ALSBaseline(lambda_=1e5, p=0.001)),
        ("savgol", SavitzkyGolaySmoother(window_length=9, polyorder=3)),
        ("norm", VectorNormalizer(norm="l2")),
    ]
)
X_proc = pipe.fit_transform(fs.x)

# 3. Run PCA
_, pca_res = run_pca(X_proc, n_components=3)
print("Explained variance ratio:", pca_res.explained_variance_ratio_)
```

### Next steps

- See [Libraries](libraries.md) to build HDF5 spectral libraries and use public datasets.
- See [Validation & chemometrics](validation_chemometrics_oils.md) for oil-authentication examples.
- Explore the CLI workflows from the command line with `foodspec about` and related commands.
