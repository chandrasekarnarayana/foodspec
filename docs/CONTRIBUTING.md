# Contributing to FoodSpec

## Dev setup
- Python >=3.10, recommended `pip install -e .[dev]`
- Run tests: `pytest`
- Lint: `ruff check`

## Adding features
- Keep core logic in `src/foodspec`. User interfaces are CLI-only.
- Add tests for new functionality (unit + small integration).
- Update docs when adding protocols, steps, or config fields.

## Protocols
- Protocols live in `examples/protocols/` (YAML/JSON).
- Include `name`, `version`, `steps`, `expected_columns`, and (optionally) `min_foodspec_version`.

## HDF5 schema
- SpectralDataset/HyperspectralDataset store instrument metadata, preprocessing history, and provenance.
- Keep schema version in sync if fields change.

## Docs
- Architecture: `docs/architecture.md`
- Protocol cookbook: `docs/protocol_cookbook.md`
- HSI/harmonization: `docs/hsi_and_harmonization.md`
