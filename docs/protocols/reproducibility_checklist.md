# Protocols: Reproducibility Checklist

**Purpose:** Ensure FoodSpec analyses are transparent, documented, and repeatable.  
**Audience:** Researchers publishing results or conducting audits.  
**Time to read:** 10–15 minutes to fill in.  
**Prerequisites:** Completed a FoodSpec analysis or QC run; have outputs saved.

---

## Interactive Checklist Table

| Category | Item | Why | Example Command |
|----------|------|-----|-----------------|
| **Data** | Provenance documented | Know where data came from | `sha256sum data/oils.csv` |
| **Data** | Format validated | Correct structure for FoodSpec | `foodspec-validate --input data/oils.csv` |
| **Data** | Metadata complete | Sample IDs, labels, batches | `head -5 data/oils.csv` |
| **Preprocess** | Baseline params logged | Reproduce exact preprocessing | `ALS: λ=1e5, p=0.01` |
| **Preprocess** | Normalization method recorded | SNV vs. Vector vs. MSC | `vector` |
| **Preprocess** | Smoothing parameters saved | Savitzky-Golay window/poly | `window=9, poly=3` |
| **Features** | Feature extraction rules saved | Know which bands/ratios used | `peaks: [1655, 1742, 1450]` |
| **Model** | Model type + params logged | Exact classifier config | `RandomForest(n_estimators=200, max_depth=10, random_state=42)` |
| **Model** | Random seed set | Deterministic results | `random_state=42` |
| **Validation** | CV strategy documented | 5-fold stratified, nested, etc. | `StratifiedKFold(n_splits=5, random_state=42)` |
| **Validation** | Metrics with uncertainty | Report std/CI, not just mean | `Accuracy: 95.2% ± 1.5%` |
| **Validation** | No data leakage | Preprocessing inside CV folds | Confirmed in code review |
| **Outputs** | Metrics saved as JSON | Machine-readable results | `metrics.json` |
| **Outputs** | Model checkpointed | Can load for prediction | `model.pkl` |
| **Outputs** | Figures generated | Visual validation | `confusion_matrix.png`, `roc_curve.png` |
| **Outputs** | Methods text exported | Reproducible prose | `methods.txt` |
| **Environment** | Package versions recorded | Know exact deps | `pip freeze > requirements.txt` |
| **Environment** | Python version logged | For compatibility | `python --version` |
| **Environment** | OS documented | Linux/macOS/Windows | Linux (Ubuntu 22.04) |
| **Artifacts** | Everything in run folder | Single source of truth | `/runs/oil_auth_20260106_120530/` |
| **Artifacts** | Git commit hash included | Track code version | `git rev-parse HEAD` |

---

## Metadata Capture Code Example

```python
import json
import sys
import subprocess
from foodspec import __version__
import numpy as np

# Capture reproducibility metadata
metadata = {
    "title": "Oil authentication via Raman spectroscopy",
    "date_run": "2026-01-06T12:03:30",
    
    # Data
    "data": {
        "source": "examples/data/oils.csv",
        "hash_sha256": "abcd1234...",  # sha256sum of input file
        "n_samples": 96,
        "n_wavenumbers": 4096,
        "labels": ["Olive", "Palm", "Sunflower", "Coconut"]
    },
    
    # Preprocessing
    "preprocessing": {
        "baseline": {"method": "als", "lambda": 1e5, "p": 0.01},
        "smoothing": {"method": "savitzky_golay", "window": 9, "poly_order": 3},
        "normalization": {"method": "vector"}
    },
    
    # Model
    "model": {
        "type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    
    # Validation
    "validation": {
        "strategy": "stratified_k_fold",
        "n_splits": 5,
        "random_state": 42
    },
    
    # Results
    "metrics": {
        "accuracy": 0.952,
        "balanced_accuracy": 0.948,
        "macro_f1": 0.948
    },
    
    # Environment
    "environment": {
        "foodspec_version": __version__,
        "python_version": sys.version,
        "numpy_version": np.__version__
    },
    
    # Code tracking
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
}

# Save with run
with open("run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Reproducibility metadata saved to run_metadata.json")
```

**Output:**
```json
{
  "title": "Oil authentication via Raman spectroscopy",
  "date_run": "2026-01-06T12:03:30",
  "data": {
    "source": "examples/data/oils.csv",
    "hash_sha256": "abcd1234...",
    "n_samples": 96,
    "n_wavenumbers": 4096
  },
  "metrics": {
    "accuracy": 0.952,
    "balanced_accuracy": 0.948
  },
  ...
}
```

---

## Run Folder Structure (Best Practice)

```
runs/oil_auth_20260106_120530/
├── run_metadata.json          # ← All reproducibility info
├── metrics.json               # ← Accuracy, F1, etc.
├── protocol.yaml              # ← YAML config used
├── requirements.txt           # ← pip freeze output
├── model.pkl                  # ← Trained classifier
├── preprocessing_pipeline.pkl # ← Scaler, baseline params
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pca_scores.png
├── methods.txt                # ← Human-readable text
├── report.html                # ← Full HTML report
└── README.md                  # ← Quick summary
```

---

## Git Workflow for Protocols

Track your protocol files in version control:

```bash
# Create protocols directory
mkdir -p protocols
git add protocols/

# Create protocol
vim protocols/oil_auth_v1.yaml

# Commit with message
git commit -m "Add oil authentication protocol (stratified 5-fold CV, ALS baseline)"

# Tag stable versions
git tag -a v1.0 -m "First stable protocol version"

# Run via Git reference
foodspec-run-protocol \
  --protocol protocols/oil_auth_v1.yaml \
  --protocol-version v1.0  # ← Can trace back to exact commit
```

---

## Checklist (record/attach)
- **Data provenance:**
  - Dataset name/version; source/DOI.
  - File format (CSV → HDF5); wavenumber ordering (ascending cm⁻¹).
  - Metadata columns (sample_id, labels, conditions).
- **Preprocessing:**
  - Baseline method + parameters (ALS λ, p).
  - Smoothing (Savitzky–Golay window/poly), derivatives if used.
  - Normalization (L2/area/internal-peak), scatter correction (SNV/MSC), cropping ranges.
  - Modality corrections (ATR, atmospheric, cosmic ray removal).
- **Features:**
  - Expected peaks/ratios; band integration ranges; fingerprint similarity settings.
- **Models & validation:**
  - Model type + hyperparameters; seeds.
  - CV design (stratified folds, groups/batches); metrics reported.
  - Thresholds for QC/novelty if applicable.
- **Artifacts:**
  - metrics.json, run_metadata.json, report.md, plots.
  - Model registry entries (path, version, foodspec_version).
  - Plot/report flags used (e.g., pca_scores, confusion_matrix, feature_importances, spectra overlays; summary_json, markdown_report, run_metadata export).
- **Environment:**
  - Python/OS; package versions; CLI command/config used.
  - Hardware notes (GPU/CPU not required here, but note if used).
- **Statistics:**
  - Tests run (e.g., ANOVA, t-test, correlation), alpha level.
  - Effect sizes reported (Cohen’s d, eta-squared).
  - Design summary (group sizes, replication, randomization).
- **Citation & licensing:**
  - Software citation (CITATION.cff); dataset licenses/DOIs.

## Example snippet (fill in per study)
- Instrument: Raman, 785 nm, resolution 4 cm⁻¹; silicon calibration daily.
- Samples: oils (olive/sunflower), 10 spectra/class; randomized acquisition.
- Preprocessing: ALS (λ=1e5, p=0.01), Savitzky–Golay (win=9, poly=3), L2 norm, crop 600–1800 cm⁻¹.
- Features: peaks 1655/1742/1450, ratios 1655/1742, 1450/1655.
- Analysis: RF classifier (n_estimators=200), stratified 5-fold CV; stats: ANOVA on ratios (alpha=0.05), Tukey post-hoc; effect size: eta-squared.
- Metrics: accuracy 0.90 ± 0.02, macro F1 0.88 ± 0.03; ANOVA p < 0.01; effect size eta² = 0.45.
- Figures: confusion_matrix.png, pca_scores.png, boxplot ratios.
- Artifacts: metrics.json, run_metadata.json, report.md, configs logged; model registry entry: models/oil_rf_v1 (foodspec 0.2.0).

## Example (oil authentication, synthetic)
- Data: `libraries/oils.h5`, labels `oil_type`, ascending cm⁻¹.
- Preprocessing: ALS (λ=1e5, p=0.01), Savitzky–Golay (win=9, poly=3), L2 norm, crop 600–1800 cm⁻¹.
- Features: Peaks 1655/1742/1450 with ratios 1655/1742, 1450/1655.
- Model: Random Forest (n_estimators=200, random_state=0), Stratified 5-fold CV, metrics: accuracy, macro F1.
- Artifacts: `runs/oil_auth_demo/metrics.json`, `confusion_matrix.png`, `run_metadata.json`, `report.md`.
- Environment: Python 3.12, foodspec 0.2.0, OS Linux; CLI command recorded; seeds fixed.
- Citation: foodspec software (CITATION.cff).

## Notes
- Store checklists with run artifacts for audits.
- Prefer configs/YAML to capture parameters; avoid manual re-entry.

## Next Steps

- [Benchmarking Framework](benchmarking_framework.md) — Compare model performance across runs.
- [Reporting Guidelines](../troubleshooting/reporting_guidelines.md) — Write up your methods and results.
- [Study Design & Data Requirements](../stats/study_design_and_data_requirements.md) — Plan experiments for reproducibility.
- [Reporting guidelines](../troubleshooting/reporting_guidelines.md)
