# Getting started

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Food scientists, analytical chemists, QC engineers, and data scientists new to FoodSpec.  
**What problem does this solve?** Understanding the FoodSpec workflow from raw spectra to validated results.  
**When to use this?** After installation, when you need an overview before diving into specific workflows.  
**Why it matters?** Understanding data formats and the typical pipeline prevents common mistakes and saves time.  
**Time to complete:** 15-20 minutes  
**Prerequisites:** FoodSpec installed; basic understanding of spectroscopy concepts; familiarity with Python or CLI

---

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
from foodspec import load_library
from foodspec.validation import validate_spectrum_set
from foodspec.preprocess.baseline import ALSBaseline
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother
from foodspec.preprocess.normalization import VectorNormalizer
from foodspec.chemometrics.pca import run_pca

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
- Libraries & formats: [../user-guide/libraries.md](../user-guide/libraries.md), [../user-guide/csv_to_library.md](../user-guide/csv_to_library.md)
- Workflows: oil authentication, heating, mixture, hyperspectral, QC
- User guide: CLI reference ([../user-guide/cli.md](../user-guide/cli.md)), preprocessing ([../methods/preprocessing/baseline_correction.md](../methods/preprocessing/baseline_correction.md))
- Keyword lookup: [../reference/keyword_index.md](../reference/keyword_index.md)

See also
- [../workflows/oil_authentication.md](../workflows/authentication/oil_authentication.md)
- [../workflows/heating_quality_monitoring.md](../workflows/quality-monitoring/heating_quality_monitoring.md)
- [../user-guide/csv_to_library.md](../user-guide/csv_to_library.md)
