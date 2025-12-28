# Developer Guide – Testing & CI

How FoodSpec is tested and how CI is configured.

## Test suite

- Location: `tests/`
- Key areas:
  - Preprocessing & harmonization: `test_preprocessing_*.py`, `test_harmonization*.py`
  - HSI: `test_hsi*.py`
  - RQ/validation: `test_rq_*.py`, `test_validation_strategies.py`
  - CLIs: `test_cli_*.py`, `test_cli_batch_outputs.py`, `test_cli_predict.py`, `test_cli_plugin.py`, `test_cli_registry.py`
  - Registry/bundle: `test_registry.py`, `test_output_bundle.py`
- Run all: `pytest`
- Targeted file: `pytest tests/test_validation_strategies.py -k batch`

## Continuous Integration

- Workflow: `.github/workflows/tests.yml`
- Matrix: Python 3.10–3.12
- Steps: install deps → lint/format (if configured) → `pytest` → `mkdocs build`

## Adding new tests

- Place new files under `tests/` with descriptive names.
- Use synthetic datasets in `tests/data/` for harmonization/HSI/vendor tests.
- For CLI tests, prefer `subprocess` or `click.testing.CliRunner` and assert exit codes plus key outputs.
- Keep tests fast and focused; use markers to group optional extras.

## Release checklist (summary)

See `RELEASE_CHECKLIST.md` for the full list. Core items:

1. Bump version in `pyproject.toml`.
2. Update `CHANGELOG.md` or `docs/changelog.md`.
3. Run full test suite and docs build.
4. Tag `vX.Y.Z` and push.
5. Build and upload to PyPI/TestPyPI.

## Troubleshooting

- If doc build fails, check that new pages are added to `mkdocs.yml`.
- For plugin-related tests, verify entry points are discoverable in the test env.
