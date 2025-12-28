# User Guide – Logging & Diagnostics

FoodSpec writes a per-run `run.log` and captures metadata to help diagnose issues.

## Where logs live
- CLI: `foodspec-run-protocol` and related commands initialize a logger; when a run folder is created, logs are written to `run.log` inside that folder.

## What is logged
- Environment snapshot (OS, Python version, PID; memory if `psutil` is available).
- Validation warnings/errors, guardrails (class counts, feature:sample ratio), CV auto-tuning.
- Step-by-step execution (preprocess, harmonize, QC, RQ, publish), harmonization diagnostics, auto-publish notes.
- Errors with stack traces in `run.log` (user-facing dialogs remain concise).

## Metadata and index
- `metadata.json` captures protocol, version, seed, inputs, validation strategy, harmonization info, and logs.
- `index.json` lists tables, figures, warnings, models, validation strategy, and harmonization details.

## How to use logs
- For CLI runs: open `run.log` in the run folder to see the execution trace and any auto-adjustments or warnings.
- For debugging prediction mismatches: errors about missing features/columns will be logged; adjust preprocessing/ratios to match the frozen model.

See also: [automation.md](automation.md), [cli_guide.md](cli.md), and [Troubleshooting](../help/troubleshooting.md) for common issues. 
