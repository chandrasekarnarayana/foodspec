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
# Getting started

Questions this page answers
- Who is foodspec for?
- How do I install it (core vs deep extra)?
- How do I run a minimal Python and CLI example?
- What is the typical pipeline?

## Who is it for?
Food scientists, analytical chemists, QC engineers, and data scientists working with Raman/FTIR spectra who need reproducible preprocessing, chemometrics, and reporting.

## Installation
- Core:
  ```bash
  pip install foodspec
  ```
- Deep-learning extra (optional 1D CNN prototype):
  ```bash
  pip install "foodspec[deep]"
  ```
- Verify:
  ```bash
  foodspec about
  ```

## Minimal Python example
```python
from pathlib import Path
import matplotlib.pyplot as plt
from foodspec.data import load_library
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.chemometrics.pca import run_pca
from foodspec.validation import validate_spectrum_set

fs = load_library(Path("libraries/oils_demo.h5"))
validate_spectrum_set(fs)

X = fs.x
for step in [
    ALSBaseline(lambda_=1e5, p=0.01, max_iter=10),
    SavitzkyGolaySmoother(window_length=9, polyorder=3),
    VectorNormalizer(norm="l2"),
]:
    X = step.fit_transform(X)

_, pca_res = run_pca(X, n_components=2)
plt.scatter(pca_res.scores[:, 0], pca_res.scores[:, 1], c="steelblue")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
plt.savefig("pca_scores.png", dpi=150)
```

## Minimal CLI example (hypothetical public dataset)
1) Convert CSV (wide) to HDF5:
```bash
foodspec csv-to-library data/oils.csv libraries/oils.h5 \
  --format wide \
  --wavenumber-column wavenumber \
  --label-column oil_type \
  --modality raman
```
2) Run oil authentication:
```bash
foodspec oil-auth libraries/oils.h5 \
  --label-column oil_type \
  --output-dir runs/oils_demo
```
Outputs: metrics.json/CSV, confusion_matrix.png, report.md in a timestamped folder.

## Typical pipeline (text diagram)
Raw spectra → CSV/TXT → **HDF5 library** → Preprocess (baseline, smoothing, normalization, crop) → Features/chemometrics (peaks/ratios/PCA/PLS/models) → Metrics & reports (plots, JSON/Markdown).

## Links
- Libraries & formats: `libraries.md`, `csv_to_library.md`
- Workflows: oil authentication, heating, mixture, hyperspectral, QC
- User guide: CLI reference (`cli.md`), preprocessing (`ftir_raman_preprocessing.md`)
- Keyword lookup: `keyword_index.md`

See also
- `oil_auth_tutorial.md`
- `heating_tutorial.md`
- `csv_to_library.md`
