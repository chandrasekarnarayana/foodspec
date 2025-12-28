# First Steps (CLI)

Run a protocol end-to-end:
```bash
foodspec-run-protocol --input examples/data/oil_synthetic.csv \
  --protocol examples/protocols/EdibleOil_Classification_v1.yml \
  --output-dir runs
```
Check environment:
```bash
foodspec-run-protocol --check-env
```
Prediction:
```bash
foodspec-predict --model path/to/model_prefix --input new_data.csv --output preds.csv
```
More CLIs: see docs/cli_help.md.
# First steps (CLI)

Run your first protocol end-to-end from the command line.

## Why use CLI?
- Reproducible, automatable runs with clear bundle outputs.

## Complete example (oil discrimination)
```bash
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --protocol examples/protocols/oil_basic.yaml \
  --output-dir runs/oil_basic_demo
```
You’ll see validation, preprocessing, RQ steps, and the run folder path in the console.

### Bundle structure (expected)
```plaintext
runs/oil_basic_demo/<timestamp>/
  report.txt
  report.html
  metadata.json
  index.json
  figures/
  tables/
  run.log
  models/   # if a model was trained/frozen
```

## Publish a narrative and figure panel
```bash
foodspec-publish runs/oil_basic_demo/<timestamp> --fig-limit 6
```
Outputs a Methods-style text plus key figures; use `--include-all-figures` for supplementary material.

## Apply a frozen model (optional)
If the run saved a frozen model:
```bash
foodspec-predict \
  --input examples/data/oils.csv \
  --model runs/oil_basic_demo/<timestamp>/models/frozen_model.pkl
```
This applies the same preprocessing/feature definitions used in training.

 

## Environment check
If dependencies seem missing:
```bash
foodspec-run-protocol --check-env
```
to verify Python version and core deps.

## Troubleshooting
- **Missing columns / protocol mismatch**: ensure required columns match `expected_columns` in the protocol (use the example CSV unchanged to test).
- **Validation errors**: the CLI prints a “Validation” block; fix blocking errors, re-run. Warnings are captured in `metadata.json` and `run.log`.
