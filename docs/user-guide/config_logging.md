# Configuration & Logging

**Purpose:** Customize FoodSpec behavior via config files and enable detailed logging for debugging.

**Audience:** Advanced users and pipeline developers; data scientists troubleshooting analyses.

**Time:** 10 minutes to set up; reference during troubleshooting.

**Prerequisites:** Python environment basics; understand FoodSpec workflows.

---

FoodSpec writes a per-run `run.log` and captures metadata to help diagnose issues.

## Quick Config Example

Create `~/.foodspec/config.yaml`:

```yaml
# Preprocessing defaults
preprocessing:
  baseline:
    method: "als"
    lam: 1e6
    p: 0.01
  smoothing:
    window_length: 11
    polyorder: 3

# ML defaults
ml:
  default_cv_strategy: "stratified_kfold"
  n_splits: 5
  random_state: 42

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "/var/log/foodspec.log"
```

**Test it:**

```python
from foodspec.config import load_config
cfg = load_config()
print(f"Baseline method: {cfg['preprocessing']['baseline']['method']}")
```

## Where Logs Live
- CLI: `foodspec-run-protocol` and related commands initialize a logger; when a run folder is created, logs are written to `run.log` inside that folder.

## What is Logged
- Environment snapshot (OS, Python version, PID; memory if `psutil` is available).
- Validation warnings/errors, guardrails (class counts, feature:sample ratio), CV auto-tuning.
- Step-by-step execution (preprocess, harmonize, QC, RQ, publish), harmonization diagnostics, auto-publish notes.
- Errors with stack traces in `run.log` (user-facing dialogs remain concise).

## Metadata and Index
- `metadata.json` captures protocol, version, seed, inputs, validation strategy, harmonization info, and logs.
- `index.json` lists tables, figures, warnings, models, validation strategy, and harmonization details.

## How to Use Logs
- For CLI runs: open `run.log` in the run folder to see the execution trace and any auto-adjustments or warnings.
- For debugging prediction mismatches: errors about missing features/columns will be logged; adjust preprocessing/ratios to match the frozen model.

## Next Steps

- [CLI Help](cli_help.md) — Command-line logging options
- [Protocols & YAML](protocols_and_yaml.md) — More advanced config
- [Troubleshooting](../help/troubleshooting.md) — Debug common issues

See also: [automation.md](automation.md), [cli_guide.md](cli.md), and [Troubleshooting](../help/troubleshooting.md) for common issues. 
