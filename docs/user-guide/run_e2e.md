# Running the end-to-end CLI (`run-e2e`)

`foodspec run-e2e` wires validation, preprocessing, feature extraction, modeling, trust evaluation, visualization, and reporting into a single command. Defaults come from your protocol file; CLI flags override without mutating the protocol.

## Inputs
- A tidy CSV with spectral columns plus labels (and optional group column for LOSO).
- A protocol YAML/JSON describing expected columns and feature settings.

### Minimal protocol skeleton
```yaml
name: Example Run-E2E
version: 0.0.1
expected_columns:
  label_col: label
  group_col: stage
steps:
  - type: preprocess
    params:
      baseline_method: none
      normalization: none
  - type: rq_analysis
    params:
      model: logreg
      validation:
        scheme: loso
        folds: 2
features:
  type: peaks
  peaks:
    - name: peak_1000
      center: 1000
      window: 2.0
    - name: peak_1002
      center: 1002
      window: 2.0
task:
  type: classification
  target: label
```

## Quickstart
Run the orchestrated pipeline with CSV + protocol:
```bash
foodspec run-e2e \
  --csv tests/fixtures/tiny_oil_fixture.csv \
  --protocol /path/to/protocol.yaml \
  --outdir runs/run_e2e_demo \
  --model logreg \
  --features peaks \
  --mode research \
  --label-col label \
  --group stage \
  --seed 1
```

Key flags:
- `--scheme` forces a validation scheme (otherwise LOSO if a group column is present, else random).
- `--trust/--no-trust` toggles calibration + conformal evaluation (requires `predict_proba`).
- `--viz/--no-viz` toggles quick confusion-matrix plotting.
- `--report/--no-report` controls HTML report generation; `--pdf` attempts a PDF export when dependencies allow.
- `--unsafe-random-cv` lets you opt into random CV even when groups exist (otherwise LOSO is enforced).

## Outputs
The command writes a run directory containing:
- `manifest.json`, `run_summary.json` (inputs, status, seed, paths)
- `metrics.csv`, `metrics_per_fold.csv`, `predictions.csv`, `features.csv`, `feature_info.json`
- `trust/evaluation.json`, `trust/calibration.json`, `trust/calibration_metrics.csv` when trust is enabled
- `plots/viz/confusion_matrix.png` when visualization is enabled
- `report.html` (and `report.pdf` when requested) plus experiment cards (`card.json`, `card.md`)

These artifacts can be reused by `foodspec report-run` or downstream publishing flows.
