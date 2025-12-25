## CLI help and defaults

- `foodspec-run-protocol --check-env` prints dependency status.
- If required args are missing, CLIs emit a short hint to see quickstart_protocol.md for expected columns.
- Common errors (missing model/protocol version) surface as readable messages.

### Protocol Runner: Parameter Overrides

The protocol runner (`foodspec-run-protocol`) supports common preprocessing and modeling overrides:
- `--seed N`: set random seed
- `--cv-folds N`: override CV splits in RQ/modeling steps
- `--normalization-mode MODE`: override preprocess normalization (e.g., `reference`, `vector`, `area`, `max`)
- `--baseline-method METHOD`: override preprocess baseline (`als`, `rubberband`, `polynomial`, `none`)
- Spike toggles: see dedicated subsection below

Notes
- Overrides apply to matching step types inside the protocol definition (e.g., `preprocess.params.normalization`).
- Smoothing parameters are controlled via the protocol file (e.g., `preprocess.params.smoothing_window` and `smoothing_polyorder`). The CLI does not add separate smoothing flags.

### Spike / Cosmic-ray Removal (Protocol Runner)

Two mutually exclusive flags control spike (cosmic-ray) removal in the preprocessing step:
- `--spike-removal`: enable spike removal
- `--no-spike-removal`: disable spike removal

Behavior
- Default: If neither flag is set, the runner uses the protocol’s configured value. If the protocol omits it, preprocessing defaults to spike removal enabled.
- Mapping: These flags override `preprocess.params.spike_removal` in the loaded protocol.
- Outputs affected:
	- A `spikes_removed` column is added to outputs when spike removal is enabled, reporting per-spectrum spike counts.
	- Preprocessing metadata includes the spike configuration (visible in run metadata files).

Examples

Enable spike removal:
```bash
foodspec-run-protocol \
	--input examples/data/oils.csv \
	--protocol examples/protocols/EdibleOil_Classification_v1.yml \
	--output-dir runs/oils \
	--spike-removal
```

Disable spike removal:
```bash
foodspec-run-protocol \
	--input examples/data/oils.csv \
	--protocol examples/protocols/EdibleOil_Classification_v1.yml \
	--output-dir runs/oils \
	--no-spike-removal
```

Combine baseline + normalization override with spike toggle:
```bash
foodspec-run-protocol \
	--input examples/data/oils.csv \
	--protocol examples/protocols/EdibleOil_Classification_v1.yml \
	--output-dir runs/oils \
	--baseline-method rubberband \
	--normalization-mode vector \
	--spike-removal
```

Cross-reference
- See protocol runner synopsis in [docs/cli.md](cli.md) (section: protocol-runner) and quickstart in [docs/quickstart_protocol.md](../01-getting-started/quickstart_protocol.md).
- The YAML experiment command `foodspec run-exp` executes end-to-end pipelines from `exp.yml`; it is complementary to `foodspec-run-protocol` but does not directly call it.

### Try it now (60 seconds)

Run a small protocol on example data:
```bash
foodspec-run-protocol \
	--input examples/data/oil_synthetic.csv \
	--protocol examples/protocols/EdibleOil_Classification_v1.yaml \
	--output-dir runs/quickstart
```

Toggle spike removal OFF and compare:
```bash
foodspec-run-protocol \
	--input examples/data/oil_synthetic.csv \
	--protocol examples/protocols/EdibleOil_Classification_v1.yaml \
	--output-dir runs/quickstart \
	--no-spike-removal
```
What changes:
- `metadata.json` records `preprocess.params.spike_removal` as `false` (when toggled off).
- When enabled, the processed data includes a `spikes_removed` column (per-spectrum spike counts) used downstream.

Combine baseline, normalization, and spike toggle:
```bash
foodspec-run-protocol \
	--input examples/data/oil_synthetic.csv \
	--protocol examples/protocols/EdibleOil_Classification_v1.yaml \
	--output-dir runs/quickstart \
	--baseline-method rubberband \
	--normalization-mode vector \
	--spike-removal
```

Output notes:
- Protocol runs write `metadata.json`, `report.txt`, and step tables under `tables/` in the run folder (`<output-dir>/<protocol>_<input-stem>/`).
- QC runs produce a scores table at `tables/scores.csv` in the QC report folder (see `foodspec qc`).
