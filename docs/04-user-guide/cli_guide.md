# User Guide – CLI reference

Why it matters: CLIs let you automate and reproduce analyses identically to the GUI, with clear outputs and provenance.

## foodspec-run-protocol
Purpose: run a protocol end-to-end (preprocess → harmonize → QC → RQ → output bundle).
```bash
foodspec-run-protocol --input my.csv --protocol examples/protocols/oil_basic.yaml --output-dir runs/demo
```
Key flags:
- `--input` (repeatable for multi-input)
- `--protocol` (YAML/JSON path or name if discoverable)
- `--output-dir`
- `--check-env` (print environment summary)
- Overrides (if enabled): `--baseline-method`, `--normalization-mode`, `--validation-strategy`, `--seed`, `--cv-folds`
- Batch/glob: `--input-dir` / `--glob` if supported by your version.
- Automation: `--auto` to run publish after protocol; `--report-level {summary,standard,full}` controls figure/text richness for auto-publish.
- Dry run: `--dry-run` to validate/estimate without executing.

Example (oil vs chips multi-input):
```bash
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --input examples/data/chips.csv \
  --protocol examples/protocols/oil_vs_chips.yaml \
  --output-dir runs/oil_vs_chips
```

## foodspec-predict
Purpose: apply a frozen model (with embedded preprocessing/features) to new data.
```bash
foodspec-predict --input my.csv --model runs/demo/<ts>/models/frozen_model.pkl --output predictions.csv
# Batch mode
foodspec-predict --input-dir data/new_batch --glob "*.csv" --model runs/demo/<ts>/models/frozen_model.pkl --output-dir preds/
```
Supports `--check-env` to inspect dependencies.

## foodspec-publish
Purpose: generate a methods-style narrative and figure panels from a run folder.
```bash
foodspec-publish runs/demo/<ts> --fig-limit 6
foodspec-publish runs/demo/<ts> --profile qa      # quick figure selection for QA
foodspec-publish runs/demo/<ts> --profile publication --include-all-figures
```
Flags: `--include-all-figures` (export everything), `--fig-limit N`.

## foodspec-registry
Purpose: list/query registered runs/models (provenance).
```bash
foodspec-registry list
foodspec-registry query --protocol oil_basic
```
Registry stores protocol name/version, dataset hash, preprocessing summary, validation strategy, model paths/metrics.

## foodspec-plugin
Purpose: manage plugins (protocols, vendor loaders, harmonization strategies).
```bash
foodspec-plugin list
foodspec-plugin install my-plugin
foodspec-plugin remove my-plugin
```

## Tips
- Use `--help` on any command for full options.
- Check validation output in the console; blocking errors must be resolved before the run proceeds.
- All commands write logs/run folders; see `metadata.json` and `index.json` for details.

See also: [first-steps_cli.md](../01-getting-started/first-steps_cli.md), [cookbook_registry_reporting.md](../03-cookbook/cookbook_registry_reporting.md), and [registry_and_plugins.md](registry_and_plugins.md).
