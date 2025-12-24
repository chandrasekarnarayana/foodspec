# Quickstart (CLI)

This walkthrough shows a first end-to-end run using the foodspec command-line interface.

## 1) Prepare data
Use a small CSV (wide or long) of spectra or a public dataset you downloaded. Example wide CSV layout:
```
wavenumber,s1,s2
500,10.1,12.3
502,10.3,12.4
...
```

## 2) Convert CSV → HDF5 library
```bash
foodspec csv-to-library \
  data/oils_wide.csv \
  libraries/oils_demo.h5 \
  --format wide \
  --wavenumber-column wavenumber \
  --modality raman \
  --label-column oil_type
```
This creates an HDF5 spectral library usable by all workflows (validated `FoodSpectrumSet`).

## 3) Run oil authentication
```bash
foodspec oil-auth \
  libraries/oils_demo.h5 \
  --label-column oil_type \
  --output-dir runs/oil_demo
```
Outputs (timestamped folder):
- `metrics.json` / CSV of CV metrics
- `confusion_matrix.png`
- `report.md` / summary.json

## 4) Inspect results
- Accuracy/F1 in metrics.json
- Confusion matrix plot shows class separation
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

**See full guide**: [Multi-Modal Workflows](multimodal_workflows.md)

**Use cases:**
- ✅ Olive oil authentication (Raman confirms FTIR)
- ✅ Novelty detection (modality disagreement flags unknowns)
- ✅ Robustness validation (cross-lab/cross-technique agreement)
