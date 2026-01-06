# Quickstart (CLI)

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** QC engineers, lab technicians, anyone without Python experience  
**What problem does this solve?** Run a complete oil authentication analysis from CSV to results using command-line  
**When to use this?** First-time users, batch processing, automation scripts  
**Why it matters?** CLI workflows are scriptable, reproducible, and don't require Python knowledge  
**Time to complete:** 10 minutes  
**Prerequisites:** FoodSpec installed (`pip install foodspec`), CSV file with spectra, basic terminal knowledge

---

## Installation

### Option 1: pip (Recommended)
```bash
pip install foodspec
```

### Option 2: conda
```bash
conda install -c conda-forge foodspec
```

### Option 3: Development version (from source)
```bash
git clone https://github.com/chandrasekarnarayana/foodspec.git
cd foodspec
pip install -e .
```

**Verify installation:**
```bash
foodspec --version
# Output: 1.0.0
```

---

## Dataset Format

!!! tip "Data Format Reference"
    See [Data Format Reference](../reference/data_format.md) for complete schema specifications, unit conventions, and validation checklist. Key terms defined in [Glossary](../reference/glossary.md).

### CSV Requirements
Your input CSV must have:
1. **Wavenumber/Wavelength column** (e.g., "wavenumber", "wavelength", "cm-1")
2. **Spectra columns** — one per sample (intensity values)
3. **Label column** (optional) — class labels for classification (e.g., "oil_type", "sample_id")

### Example CSV (wide format):
```csv
wavenumber,olive_oil_1,olive_oil_2,sunflower_oil_1,palm_oil_1
1000.0,5.2,5.1,4.8,6.1
1010.0,5.5,5.3,5.0,6.3
1020.0,5.8,5.6,5.2,6.5
...
3000.0,2.1,2.0,2.3,1.9
```

### Minimal Requirements:
- **Min spectra:** 3 per class (10+ recommended)
- **Min wavenumbers:** 50 (100+ recommended)
- **Spacing:** Regular or irregular (FoodSpec handles both)
- **Units:** Any (cm⁻¹, nm, etc.) — FoodSpec doesn't care

### Add labels (CSV with metadata):
```csv
wavenumber,oil_type,batch,s1,s2,s3
1000.0,olive,A,5.2,5.1,4.8
1010.0,olive,A,5.5,5.3,5.0
...
```

---

## One Complete Example (Oil Authentication)

### Step 1: Create sample data
```bash
# Generate synthetic oil spectra for testing
python << 'EOF'
import numpy as np
import pandas as pd

# Create toy dataset
np.random.seed(42)
wavenumbers = np.linspace(1000, 3000, 150)

# Generate 10 samples per class
olive_spectra = np.random.normal(5.0, 0.3, (10, 150))
palm_spectra = np.random.normal(6.0, 0.3, (10, 150))
sunflower_spectra = np.random.normal(4.5, 0.3, (10, 150))

# Combine
X = np.vstack([olive_spectra, palm_spectra, sunflower_spectra])
labels = ['olive']*10 + ['palm']*10 + ['sunflower']*10

# Create DataFrame
df = pd.DataFrame(X, columns=[f'{w:.1f}' for w in wavenumbers])
df.insert(0, 'oil_type', labels)
df.insert(0, 'wavenumber', wavenumbers)

# Save
df.to_csv('oils_demo.csv', index=False)
print("✓ Created oils_demo.csv (30 samples, 150 wavenumbers)")
EOF
```

### Step 2: Convert CSV to FoodSpec library
```bash
foodspec csv-to-library \
  oils_demo.csv \
  oils_demo.h5 \
  --wavenumber-column wavenumber \
  --label-column oil_type \
  --modality raman
```

**Expected output:**
```plaintext
✓ Loaded 30 spectra from oils_demo.csv
✓ Wrote 30 spectra to oils_demo.h5
```

### Step 3: Run oil authentication workflow
```bash
foodspec oil-auth \
  oils_demo.h5 \
  --output-dir runs/demo
```

**Expected output:**
```yaml
✓ Loaded 30 spectra
✓ Preprocessing (ALS baseline, normalization, smoothing)
✓ Running RQ (ratiometric features) analysis
✓ Cross-validation: 5-fold
✓ Mean accuracy: 0.87 (±0.12)
✓ Results saved to runs/demo/20241228_120000_run/
```

### Step 4: View results
```bash
# Print summary report
cat runs/demo/*/report.txt

# View metrics
cat runs/demo/*/metrics.json | python -m json.tool

# List all outputs
ls -lh runs/demo/*/
```

**Expected files:**
```plaintext
report.txt              — Text summary with accuracy, key ratios
metrics.json            — Detailed CV metrics
confusion_matrix.png    — Classification visualization
rq_summary.csv          — Extracted ratiometric features
tables/                 — Additional analysis tables
```

---

## Explore Other Workflows

### Oil Heating Stability
Track degradation during frying:
```bash
foodspec heating \
  oils_demo.h5 \
  --output-dir runs/heating
```

### Mixture Analysis
Quantify oil blends:
```bash
foodspec mixture \
  oils_demo.h5 \
  --components olive,sunflower \
  --output-dir runs/mixture
```

### List all commands
```bash
foodspec --help
```

---

## Expected Outputs

### Directory Structure
```yaml
runs/demo/20241228_120000_run/
├── report.txt                 # Text summary
├── metrics.json               # Cross-validation metrics
├── confusion_matrix.png       # Classification plot
├── rq_summary.csv             # Feature matrix
├── tables/
│   ├── rq_features.csv
│   └── cv_results.csv
└── metadata.json              # Run provenance
```

### Typical Metrics (metrics.json)
```json
{
  "balanced_accuracy": 0.87,
  "f1_macro": 0.85,
  "roc_auc_ovr": 0.92,
  "n_samples": 30,
  "n_features": 12,
  "cv_fold": 5,
  "timestamp": "2024-12-28T12:00:00Z"
}
```

---

## Troubleshooting (Top 5 Issues)

### 1️⃣ **"No such command: oil-auth"**
**Cause:** FoodSpec not installed or installed incorrectly

**Fix:**
```bash
# Reinstall
pip uninstall -y foodspec && pip install foodspec

# Verify
foodspec --version
foodspec oil-auth --help
```

---

### 2️⃣ **"FileNotFoundError: oils_demo.csv not found"**
**Cause:** CSV path is wrong or file doesn't exist

**Fix:**
```bash
# Check file exists
ls -l oils_demo.csv

# Use absolute path
foodspec csv-to-library /full/path/to/oils_demo.csv oils_demo.h5

# Or run from correct directory
cd /path/to/data
foodspec csv-to-library oils_demo.csv oils_demo.h5
```

---

### 3️⃣ **"Wavenumber column not found"**
**Cause:** Column name doesn't match data

**Fix:**
```bash
# Check column names
head -1 oils_demo.csv

# Use correct name
foodspec csv-to-library oils_demo.csv oils_demo.h5 \
  --wavenumber-column "cm-1"  # or whatever column name exists

# If no wavenumber column, generate one
python << 'EOF'
import pandas as pd
df = pd.read_csv('oils_demo.csv')
df.insert(0, 'wavenumber', range(len(df)))
df.to_csv('oils_demo_fixed.csv', index=False)
EOF
```

---

### 4️⃣ **"Error: expected 4 arguments, got 1"**
**Cause:** Missing required arguments

**Fix:**
```bash
# Always provide: input, output, wavenumber-column
foodspec csv-to-library \
  oils_demo.csv oils_demo.h5 \
  --wavenumber-column wavenumber

# Or use --help to see required args
foodspec csv-to-library --help
```

---

### 5️⃣ **"Out of memory" or "Empty results"**
**Cause:** Dataset too large or no valid spectra

**Fix:**
```bash
# Check file size and number of spectra
wc -l oils_demo.csv

# For large files, subsample
python << 'EOF'
import pandas as pd
df = pd.read_csv('oils_demo.csv')
df_small = df.iloc[::10, :]  # every 10th row
df_small.to_csv('oils_demo_small.csv', index=False)
print(f"Reduced to {len(df_small)} spectra")
EOF

# Run on smaller subset
foodspec csv-to-library oils_demo_small.csv oils_demo_small.h5
```

---

## Copy-Paste Quick Reference

### Create synthetic data
```bash
python << 'EOF'
import numpy as np
import pandas as pd
np.random.seed(42)
w = np.linspace(1000, 3000, 150)
df = pd.DataFrame(
    np.vstack([np.random.normal(5, 0.3, (10, 150)),
               np.random.normal(6, 0.3, (10, 150))]),
    columns=[f'{x:.1f}' for x in w]
)
df.insert(0, 'oil_type', ['olive']*10 + ['palm']*10)
df.insert(0, 'wavenumber', w)
df.to_csv('oils_demo.csv', index=False)
print("✓ oils_demo.csv created")
EOF
```

### Full pipeline
```bash
foodspec csv-to-library oils_demo.csv oils_demo.h5 --wavenumber-column wavenumber --label-column oil_type --modality raman && \
foodspec oil-auth oils_demo.h5 --output-dir runs/demo && \
cat runs/demo/*/report.txt
```

---

## Additional Resources

- **[Data Format Reference](../reference/data_format.md)** - Data validation checklist, schema formats
- **[Glossary](../reference/glossary.md)** - Terminology (wavenumber, baseline, CV strategy, leakage)
- **[CLI Help](../user-guide/cli_help.md)** - Complete CLI command documentation
- **[Workflows](../workflows/index.md)** - Ready-to-use analysis protocols

---

## Next Steps

- ✅ Run `foodspec oil-auth --help` to explore options
- ✅ Try `foodspec heating` or `foodspec mixture` for other analyses
- ✅ Switch to Python API for custom workflows: [Python Quickstart](quickstart_python.md)
- ✅ Deep dive: [CLI Guide](../user-guide/cli.md)
- report.md summarizes run parameters and files

Tips:
- Use `--classifier-name` to switch models (rf, svm_rbf, logreg, etc.).
- Add `--save-model` to persist the fitted pipeline via the model registry.
- For long/tidy CSVs, use `--format long --sample-id-column ... --intensity-column ...`.

## Run from exp.yml (one command)

Define everything in a single YAML file `exp.yml`:
```yaml
dataset:
  path: data/oils_demo.h5
  modality: raman
  schema:
    label_column: oil_type
preprocessing:
  preset: standard
qc:
  method: robust_z
  thresholds:
    outlier_rate: 0.1
features:
  preset: specs
  specs:
    - name: band_1
      ftype: band
      regions:
        - [1000, 1100]
modeling:
  suite:
    - algorithm: rf
      params:
        n_estimators: 50
reporting:
  targets: [metrics, diagnostics]
outputs:
  base_dir: runs/oils_exp
```

Run it end-to-end:
```bash
foodspec run-exp exp.yml
# Dry-run (validate + hashes only)
foodspec run-exp exp.yml --dry-run
 # Emit single-file artifact for deployment
 foodspec run-exp exp.yml --artifact-path runs/oils_exp.foodspec
```
The command builds a RunRecord (config/dataset/step hashes, seeds, environment), executes QC → preprocess → features → train, and exports metrics/diagnostics/artifacts + provenance to `base_dir`.

## Temporal & Shelf-life (CLI)

### Aging (degradation trajectories + stages)
```bash
foodspec aging \
  libraries/time_series_demo.h5 \
  --value-col degrade_index \
  --method linear \
  --time-col time \
  --entity-col sample_id \
  --output-dir runs/aging_demo
```
Outputs: `aging_metrics.csv`, `stages.csv`, and a sample fit figure under a timestamped folder.

### Shelf-life (remaining time to threshold)
```bash
foodspec shelf-life \
  libraries/time_series_demo.h5 \
  --value-col degrade_index \
  --threshold 2.0 \
  --time-col time \
  --entity-col sample_id \
  --output-dir runs/shelf_life_demo
```
Outputs: `shelf_life_estimates.csv` with `t_star`, `ci_low`, `ci_high` per entity, plus a quick-look figure.

## Multi-Modal & Cross-Technique Analysis (Python API)

FoodSpec supports **multi-modal spectroscopy** (Raman + FTIR + NIR) for enhanced authentication and cross-validation. While there's no dedicated CLI command yet, the Python API enables powerful multi-modal workflows:

### Quick Example

```python
from foodspec.core import FoodSpectrumSet, MultiModalDataset
from foodspec.ml.fusion import late_fusion_concat, decision_fusion_vote
from foodspec.stats.fusion_metrics import modality_agreement_kappa
from sklearn.ensemble import RandomForestClassifier

# Load aligned datasets (same samples, different techniques)
raman = FoodSpectrumSet.from_hdf5("olive_raman.h5")
ftir = FoodSpectrumSet.from_hdf5("olive_ftir.h5")
mmd = MultiModalDataset.from_datasets({"raman": raman, "ftir": ftir})

# **Late fusion**: Concatenate features, train joint model
features = mmd.to_feature_dict()
result = late_fusion_concat(features)
X_fused = result.X_fused
y = raman.sample_table["authentic"]

clf = RandomForestClassifier()
clf.fit(X_fused, y)
y_pred = clf.predict(X_fused)

# **Decision fusion**: Train separate models, combine predictions
predictions = {}
for mod, ds in mmd.datasets.items():
    clf = RandomForestClassifier()
    clf.fit(ds.X, ds.sample_table["authentic"])
    predictions[mod] = clf.predict(ds.X)

# Majority voting
vote_result = decision_fusion_vote(predictions, strategy="majority")

# **Agreement metrics**: Check cross-technique consistency
kappa_df = modality_agreement_kappa(predictions)
print(kappa_df)  # Cohen's kappa matrix (κ > 0.8 = excellent agreement)
```

**See full guide**: [Multi-Modal Workflows](../05-advanced-topics/multimodal_workflows.md)

**Use cases:**
- ✅ Olive oil authentication (Raman confirms FTIR)
- ✅ Novelty detection (modality disagreement flags unknowns)
- ✅ Robustness validation (cross-lab/cross-technique agreement)

---

## Need Help?

- **Installation errors, NaNs, shape mismatches?** → [Troubleshooting Guide](../10-help/troubleshooting.md)
- **Questions about methods or usage?** → [FAQ](../10-help/faq.md)
- **Report a bug:** [GitHub Issues](https://github.com/chandrasekarnarayana/foodspec/issues)
