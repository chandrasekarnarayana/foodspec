# Developer Guide – Contributing

This page distills the key contributor practices for FoodSpec. For the canonical rules, see the top-level [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Quick links
- Contribution guide: `CONTRIBUTING.md`
- Coding style: defined in `pyproject.toml` (check lint/format settings). Follow readable, well-documented code with minimal external deps.
- Testing & CI: see `06-developer-guide/testing_and_ci.md`

## Local checklist (abridged)
1) Create a venv and install dev extras:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -e ".[dev]"
```
2) Run lint/format (if configured) and tests:
```bash
pytest                  # full suite
```
3) Build docs (optional but recommended for doc changes):
```bash
mkdocs build
```
4) Update docs when you change protocols, steps, or CLIs (see `docs/04-user-guide/` and `docs/05-advanced-topics/`).
5) Open a PR with a concise summary, link related issues, and note any new protocols/plugins or breaking changes.

## Selected test targets
- Core logic: `tests/test_preprocessing_*.py`, `tests/test_rq_*.py`
- Harmonization/HSI: `tests/test_harmonization*.py`, `tests/test_hsi*.py`
- Validation: `tests/test_validation_strategies.py`
- CLIs: `tests/test_cli_*.py`
- Registry/plugins/bundle: `tests/test_registry.py`, `tests/test_output_bundle.py`, `tests/test_cli_plugin.py`
 

Run specific files with `pytest tests/test_validation_strategies.py -k batch`.

## Filing issues
- Use the GitHub issue templates (Bug/Feature). Include OS, Python version, command invoked, protocol name, and minimal data if possible.

## Releases
- Follow `RELEASE_CHECKLIST.md` and ensure CI is green before tagging and publishing to PyPI/TestPyPI.
