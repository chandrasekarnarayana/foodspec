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
```
The command builds a RunRecord (config/dataset/step hashes, seeds, environment), executes QC → preprocess → features → train, and exports metrics/diagnostics/artifacts + provenance to `base_dir`.
