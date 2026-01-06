# Getting Started

<!-- CONTEXT BLOCK (mandatory) -->
**Purpose:** Understand FoodSpec's workflow and choose your path (Python vs. CLI).  
**Audience:** Complete beginners; no spectroscopy background required.  
**Time:** 5-10 minutes.  
**Prerequisites:** FoodSpec installed; basic Python or terminal knowledge.

---

## The 30-Second Version

```python
# The absolute minimal example
from foodspec import __version__
print(f"FoodSpec {__version__} is ready!")

# Load some oil data
from foodspec.io import load_csv_spectra
spectra = load_csv_spectra("examples/data/oils.csv")
print(f"Loaded {len(spectra)} spectra")
```

**Expected output:**
```
FoodSpec 1.0.0 is ready!
Loaded 96 spectra
```

---

## Choose Your Path

### Path 1: Python API (Interactive, Customizable)

Best for: Learning, experimentation, custom analysis

**Quickstart:**
1. [15-Minute Quickstart](quickstart_15min.md) — Get working code now
2. [Oil Authentication](../workflows/authentication/oil_authentication.md) — Real example
3. [Full End-to-End](../workflows/end_to_end_pipeline.md) — Every step explained

**Key modules:**
```python
from foodspec.datasets import load_oil_example_data
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv
```

---

### Path 2: CLI (Reproducible, Automatable)

Best for: Reproducibility, batch processing, production use

**Quickstart:**
1. [First Steps (CLI)](first-steps_cli.md) — Run your first command
2. [Protocol Design](../workflows/domain_templates.md) — Create YAML configs
3. [Reproducibility](../reproducibility.md) — Best practices

**Key commands:**
```bash
foodspec --version                    # Verify installation
foodspec-run-protocol \               # Run analysis
  --protocol myprotocol.yaml \
  --input data.csv \
  --output-dir results
```

---

## Who is it for?

Food scientists, analytical chemists, QA engineers, and data scientists working with Raman/FTIR spectra who need reproducible preprocessing, chemometrics, and reporting.

---

## What Does FoodSpec Do?

**In:** Spectral data (CSV or HDF5) + metadata (labels, batch info)  
**Out:** Validated model + metrics + figures + reproducibility report

**Typical workflow:**

```
Raw spectra (instrument file)
        ↓
  CSV → HDF5 (standardized format)
        ↓
   Preprocess (baseline, smooth, normalize)
        ↓
   Extract features (peaks, ratios, PCA)
        ↓
   Train & validate (cross-validation)
        ↓
   Results (metrics, figures, JSON report)
```

---

## Installation Options

**Core (always install first):**
```bash
pip install foodspec
```

**Deep learning (optional, for 1D CNN):**
```bash
pip install "foodspec[deep]"
```

**Verify installation:**
```bash
foodspec --version  # Should print version
foodspec about      # Detailed info
```

---

## Data Format Quick Check

FoodSpec expects data in one of these formats:

### CSV (Simplest)

```csv
sample_id,wavenumber,intensity,label
OO_001,4000.0,0.234,Olive
OO_001,3998.0,0.235,Olive
OO_002,4000.0,0.241,Olive
```

**Load with:**
```python
from foodspec.io import load_csv_spectra
spectra = load_csv_spectra("oils.csv", label_column="label")
```

### HDF5 (Efficient for large datasets)

```python
from foodspec.io import load_hdf5_library
spectra = load_hdf5_library("oils_library.h5")
```

See [Data Format Reference](../reference/data_format.md) for full details.

---

## Next Steps

**Never used FoodSpec before?**
→ Start with [15-Minute Quickstart](quickstart_15min.md)

**Prefer command-line?**
→ Go to [First Steps (CLI)](first-steps_cli.md)

**Want to understand reproducibility?**
→ Read [Reproducibility Guide](../reproducibility.md)

**Ready for real examples?**
→ [Oil Authentication](../workflows/authentication/oil_authentication.md)

---

## FAQ (Quick Answers)

**Q: Can I use my own data?**  
A: Yes! See [Data Format Reference](../reference/data_format.md) for import instructions.

**Q: Do I need to know Python?**  
A: No! Use CLI with YAML configs. Or yes, use Python API for flexibility.

**Q: How long does an analysis take?**  
A: Usually < 1 minute for 100 samples. Depends on data size and model.

**Q: Can I reproduce old analyses?**  
A: Yes! Save protocols and metadata with each run. See [Reproducibility](../reproducibility.md).

---

## Questions this page answers
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
