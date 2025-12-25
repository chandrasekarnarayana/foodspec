# CI/CD Pipeline & Local Development Guide

This document describes the hardened GitHub Actions CI/CD pipeline and how to run the same checks locally.

## Overview

The CI pipeline consists of **5 parallel job groups**:

1. **Lint** (5 min) - Ruff check & format validation
2. **Test Matrix** (3×20 min) - Python 3.10, 3.11, 3.12 with coverage & artifacts
3. **Packaging** (10 min) - Build sdist/wheel, validate metadata, test import
4. **Docs** (10 min) - MkDocs strict build with broken link detection
5. **Optional Deps** (20 min) - Test with ML extras (XGBoost, LightGBM)

**Gating**: All must pass. Lint fails fast; tests run in parallel.

---

## Quick Start

### Install Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with all dev dependencies
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,ml]"
```

### Run All Local Checks (5 minutes)

```bash
# Lint
ruff check src tests scripts
ruff format --check src tests scripts

# Tests with coverage
export MPLBACKEND=Agg PYTHONPATH=$PWD/src
pytest --cov=foodspec --cov-report=term-missing tests

# Packaging
python -m build
twine check dist/*
pip install dist/foodspec-*.whl
python -c "import foodspec; print(f'foodspec {foodspec.__version__}')"

# Docs
mkdocs build --strict --site-dir site

# Line count check (optional)
python tools/check_linecount.py --max-lines 600
```

---

## Detailed CI Jobs

### 1. Lint Job

**When**: Runs first on push/PR; fails fast to save CI time.

**What it checks**:
- Code style (E/F/W rules)
- Import sorting (I)
- Code formatting

**Local equivalent**:
```bash
ruff check src tests scripts
ruff format --check src tests scripts
```

**Fix automatically**:
```bash
ruff check --fix src tests scripts
ruff format src tests scripts
```

---

### 2. Test Job (Matrix: 3.10, 3.11, 3.12)

**When**: Runs after lint passes; parallelized across Python versions.

**What it checks**:
- All pytest tests pass
- Coverage ≥75% (enforced)
- Imports work correctly in src-layout
- No matplotlib headless failures (MPLBACKEND=Agg)

**Local equivalent**:
```bash
export MPLBACKEND=Agg PYTHONPATH=$PWD/src

# Run all tests with coverage
pytest \
  --cov=foodspec \
  --cov-report=term-missing \
  --cov-report=html \
  --tb=short \
  tests

# Run specific test
pytest tests/core/test_import.py -v

# Run with timeout (matches CI 30min max)
pytest --timeout=1800 tests
```

**Environment variables**:
- `MPLBACKEND=Agg` - Headless matplotlib (no display server)
- `PYTHONPATH=$PWD/src` - Ensure src-layout imports work

**Artifacts generated**:
- `coverage.xml` (codecov upload if Python 3.11)
- `coverage.html` (local HTML report)
- `junit-3.11.xml` (test results)

---

### 3. Packaging Job

**When**: Runs after lint passes; validates build system.

**What it checks**:
- sdist builds without errors
- wheel builds without errors
- Package metadata is valid (README rendering, version, etc.)
- Can import foodspec from wheel

**Local equivalent**:
```bash
# Build
python -m build

# Validate metadata
twine check dist/*

# Test install & import
pip install dist/foodspec-*.whl
python -c "import foodspec; print(f'foodspec {foodspec.__version__}')"
```

**Common issues**:
- Missing `__init__.py` in packages
- `version` not set in `pyproject.toml`
- README not found or malformed
- Invalid classifiers

---

### 4. Docs Job

**When**: Runs after lint passes; validates documentation build.

**What it checks**:
- MkDocs builds without errors
- No broken internal links (strict mode)
- All code examples are correct

**Local equivalent**:
```bash
# Build in strict mode (fails on any link errors)
mkdocs build --clean --strict --site-dir site

# Serve for preview (development)
mkdocs serve  # http://localhost:8000
```

**mkdocs.yml already configured**:
- Material theme
- mkdocstrings for Python API docs
- git-revision-date plugin
- Strict link checking enabled

---

### 5. Test Optional Dependencies Job

**When**: Runs after lint passes; ensures optional extras work.

**What it checks**:
- Smoke tests pass with `[dev,ml]` extras
- XGBoost and LightGBM don't break import

**Local equivalent**:
```bash
pip install -e ".[dev,ml]"
pytest tests/core/test_import.py -v
```

---

## GitHub Actions Improvements

### A. Workflow Structure ✓

- **Separate jobs**: lint → fast-fail; test/packaging/docs in parallel
- **Concurrency**: Cancels redundant runs on force-push
- **Triggers**: push, pull_request, workflow_dispatch
- **Permissions**: Minimal per job (read by default)
- **Timeouts**: 10-30 min per job (prevents hung runners)

### B. Python Setup & Caching ✓

- **Cache strategy**: `setup-python` with pip cache by `pyproject.toml` hash
- **PYTHONPATH**: Explicitly set in tests for src-layout
- **MPLBACKEND**: Set to `Agg` for headless matplotlib
- **Editable install**: `pip install -e ".[dev]"` in all jobs

### C. Linting & Formatting ✓

- **Ruff only**: Removed black (ruff format is equivalent)
- **Config**: Consolidated in `pyproject.toml`
- **Lint job runs first**: Saves time on syntax errors

### D. Testing ✓

- **Coverage enforced**: 75% minimum (via pytest.ini)
- **Test artifacts**: junit XML uploaded per Python version
- **Matrix**: Python 3.10, 3.11, 3.12 in parallel
- **Codecov**: Optional upload (codecov action included)

### E. Packaging & Installation ✓

- **Build check**: `python -m build` validates sdist+wheel
- **Metadata validation**: `twine check dist/*`
- **Import test**: Validates wheel is installable
- **Separate job**: Runs in parallel with tests

### F. Docs Build ✓

- **Strict mode**: `mkdocs build --strict` catches broken links
- **MkDocs extras**: All installed via `[dev]`
- **Artifacts**: Site uploaded to GitHub Pages

### G. Common Failure Anticipation ✓

| Issue | Solution |
|-------|----------|
| Headless matplotlib | `MPLBACKEND=Agg` in test env |
| Import errors (src-layout) | `PYTHONPATH=$PWD/src` set |
| Missing optional deps | Graceful skip or separate job |
| Broken docs links | `mkdocs build --strict` |
| Cache misses | Use `setup-python` with `cache: pip` |

### H. Quality Gates ✓

- **Ruff**: Check + Format enforced
- **Coverage**: 75% minimum (can adjust via `pytest.ini`)
- **Tests**: All must pass (fail-fast on lint)
- **Docs**: Strict build (no broken links)
- **Line count**: Optional check available via `tools/check_linecount.py`

---

## Pre-commit Hook Setup (Optional)

To catch issues **before** CI:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks from .pre-commit-config.yaml
pre-commit install

# Run on all files
pre-commit run --all-files
```

`.pre-commit-config.yaml` already includes:
- ruff check & format
- trailing whitespace, EOF fixers
- YAML validation
- Large file detection (>1MB)
- Merge conflict detection

---

## Troubleshooting CI Failures

### Lint fails: "Ruff check failed"

```bash
ruff check --fix src tests scripts
ruff format src tests scripts
git add -A && git commit -m "ci: lint fixes"
```

### Tests fail: "Coverage below 75%"

```bash
pytest --cov=foodspec --cov-report=html tests
# Open htmlcov/index.html to see what's not covered
# Add tests or adjust omit list in pyproject.toml
```

### Docs fail: "broken link"

```bash
mkdocs build --strict
# mkdocs will report the broken link
# Fix the link in the .md file
```

### Packaging fails: "Invalid metadata"

```bash
python -m build
twine check dist/*  # Shows what's wrong
# Common: invalid classifiers, missing version, bad README markup
```

### Tests timeout or hang

- Check if tests download external data (should mock or use local fixtures)
- Check for infinite loops in test setup
- Increase timeout (edit workflow if >30min is needed)

---

## Performance Tips

1. **Parallel CI**: Tests run on Python 3.10/3.11/3.12 in parallel (~20 min total)
2. **Caching**: Pip cache shared across jobs for same Python version
3. **Lint fast**: Ruff is ~100x faster than black + mypy
4. **Early exit**: Lint runs first and fails fast
5. **Separate jobs**: Packaging and docs don't block each other

---

## Configuration Reference

### pyproject.toml

```ini
[tool.ruff]
line-length = 120
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.ruff.format]
line-length = 120

[tool.pytest.ini_options]
addopts = "--cov=foodspec --cov-fail-under=75"
testpaths = ["tests"]
pythonpath = ["src"]
```

### .github/workflows/ci.yml

- **Lint**: Fast path, fails fast
- **Test**: Parallelized matrix, uploads coverage
- **Packaging**: Build validation
- **Docs**: Strict link checking
- **All**: Coverage gate at 75%

---

## Next Steps

- **Add to CI**: Line count checker (optional):
  ```yaml
  - name: Check file line counts
    run: python tools/check_linecount.py --max-lines 600
  ```

- **Add to CI**: Type checking (optional, if mypy/pyright added):
  ```yaml
  - name: Type check
    run: mypy src --ignore-missing-imports
  ```

- **Codecov**: Enable on main branch for coverage tracking
  ```bash
  gh secret set CODECOV_TOKEN --body "$TOKEN"
  ```

---

## Questions?

See `.github/workflows/ci.yml` for the full pipeline definition.
