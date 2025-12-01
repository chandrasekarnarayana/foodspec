# Configuration & logging

## Config files
- Supported formats: YAML (`.yml/.yaml`) and JSON (`.json`).
- Load with `load_config(path)`; CLI commands accept `--config` to apply a base config.
- CLI overrides: values passed on the command line overwrite config entries (via `merge_cli_overrides`).
- Typical keys: `input_hdf5`, `output_dir`, `label_column`, `classifier_name`, `cv_splits`.

## Logging
- foodspec uses lightweight logging to record environment info (versions, platform, timestamp).
- Run metadata may be written alongside reports (e.g., `run_metadata.json`).
- If you want more verbosity, configure Python logging before calling foodspec APIs.
- When debugging, check the timestamped run directory for summary.json/metrics.json and logs emitted to stdout/stderr by CLI commands.
