# Getting started

Questions this page answers
- Who is foodspec for?
- How do I install it (core vs deep extra)?
- What data formats and metadata does it expect?
- What is the typical pipeline and why?
- Where do I go for full Python/CLI quickstarts?

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

## Data formats and metadata
- **Instrument exports**: commercial Raman/FTIR instruments often export per-spectrum TXT/CSV files (wavenumber/intensity columns) or wide CSVs (one column per spectrum).  
- **FoodSpec standard**: convert to a validated `FoodSpectrumSet` (HDF5 library) with:
  - `x`: spectra matrix (n_samples × n_wavenumbers)
  - `wavenumbers`: monotonic axis (cm⁻¹)
  - `metadata`: one row per sample (e.g., `oil_type`, `meat_type`, `species`, `heating_time`)
  - `modality`: `raman`/`ftir`/`nir`
- **Why this protocol?** Keeps spectra + metadata together, enables reproducible preprocessing/models, and matches downstream workflows (oil-auth, heating, QC).

## Typical pipeline (text diagram)
Raw spectra (instrument CSV/TXT) → **CSV→HDF5 library** → Preprocess (baseline, smoothing, normalization, crop) → Features/chemometrics (peaks/ratios/PCA/PLS/models) → Metrics & reports (plots, JSON/Markdown).

## Minimal examples (stepwise)
For full code, see the dedicated quickstarts. Highlights:

### Python (steps)
1) Load library & validate.
2) Apply simple preprocessing (ALS baseline → Savitzky–Golay → Vector norm).
3) Run PCA for a quick check.
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
for step in [ALSBaseline(lambda_=1e5, p=0.01, max_iter=10),
             SavitzkyGolaySmoother(window_length=9, polyorder=3),
             VectorNormalizer(norm="l2")]:
    X = step.fit_transform(X)

_, pca_res = run_pca(X, n_components=2)
plt.scatter(pca_res.scores[:, 0], pca_res.scores[:, 1]); plt.tight_layout()
plt.savefig("pca_scores.png", dpi=150)
```

### CLI (steps)
1) Convert CSV (wide example) to HDF5:
```bash
foodspec csv-to-library data/oils.csv libraries/oils.h5 \
  --format wide --wavenumber-column wavenumber \
  --label-column oil_type --modality raman
```
2) Run oil authentication:
```bash
foodspec oil-auth libraries/oils.h5 \
  --label-column oil_type \
  --output-dir runs/oils_demo
```
Outputs: metrics.json/CSV, confusion_matrix.png, report.md in a timestamped folder.

## Quickstarts
- Full CLI walkthrough: [quickstart_cli.md](quickstart_cli.md)
- Full Python walkthrough: [quickstart_python.md](quickstart_python.md)

## Links
- Libraries & formats: [libraries.md](libraries.md), [csv_to_library.md](csv_to_library.md)
- Workflows: oil authentication, heating, mixture, hyperspectral, QC
- User guide: CLI reference ([cli.md](cli.md)), preprocessing ([ftir_raman_preprocessing.md](ftir_raman_preprocessing.md))
- Keyword lookup: [keyword_index.md](keyword_index.md)

See also
- [workflows/oil_authentication.md](workflows/oil_authentication.md)
- [workflows/heating_quality_monitoring.md](workflows/heating_quality_monitoring.md)
- [csv_to_library.md](csv_to_library.md)
