# Testing & coverage

FoodSpec relies on extensive, fast, synthetic tests to keep the protocol trustworthy.

## Running tests
```bash
pytest --disable-warnings -q
```
With coverage (if configured):
```bash
pytest --cov=foodspec
```

## Current status
- Approximate coverage: ~91% (as of latest run).
- Scope: core data models, preprocessing, features, chemometrics, apps, CLI workflows, IO, viz, and optional deep-path guards.

## Why high coverage matters
- Protocol-oriented library: small regressions can invalidate published pipelines.
- Ensures CLI workflows (oil-auth, heating, mixture, protocol benchmarks) stay stable.
- Encourages reproducibility: deterministic, synthetic datasets avoid external dependencies.

## Testing practices
- Use `tmp_path` for filesystem; `monkeypatch` for loaders and external dependencies.
- No network or large datasets; keep tests CPU-light.
- Add tests for new public APIs and edge cases (errors as well as happy paths).
