> **Audience:** Developers and maintainers  
> This page documents internal testing practices and coverage notes; it is not required for normal users.

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

## Current status (latest run)
- Approximate coverage: **~90%** (`pytest --cov`, Python 3.12).
- Well covered: core data models, preprocessing (baseline/smoothing/normalization/derivatives), features (peaks/bands/ratios), metrics, viz, CLI workflows, protocol validation, model registry, robustness helpers.
- Partially covered but acceptable: chemometrics deep/MLP, vendor IO loaders (SPC/OPUS/JCAMP), some stats branches (hypothesis_tests/correlations edge cases), public data loaders for vendor formats.
- Gaps / future work:
  - Add mocks/fixtures to raise IO/base/core/text/vendor coverage.
  - Exercise more branches in stats (alternative tails, assumption checks) and preprocess (baseline/normalization edge cases).
  - Deep-learning paths beyond smoke tests only if DL becomes first-class.

## Why high coverage matters
- Protocol-oriented library: small regressions can invalidate published pipelines.
- Ensures CLI workflows (oil-auth, heating, mixture, protocol benchmarks) stay stable.
- Encourages reproducibility: deterministic, synthetic datasets avoid external dependencies.

## Testing practices
- Use `tmp_path` for filesystem; `monkeypatch` for loaders and external dependencies.
- No network or large datasets; keep tests CPU-light.
- Add tests for new public APIs and edge cases (errors as well as happy paths).
