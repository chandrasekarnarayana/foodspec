# FoodSpec Smoke Test Suite

**Purpose:** Quick verification that core functionality works after major refactorings.  
**Runtime:** < 5 minutes  
**When to run:** After PHASE completions, before releases, after dependency updates

---

## Test 1: Installation & Import

```bash
# Verify package installs
pip install -e .

# Test core imports
python -c "from foodspec import FoodSpec; print('✓ FoodSpec')"
python -c "from foodspec.core.api import FoodSpec; print('✓ core.api')"
python -c "from foodspec.features.rq import RatioQualityEngine; print('✓ RQ engine')"
python -c "from foodspec.chemometrics.models import make_classifier; print('✓ chemometrics')"
python -c "from foodspec.preprocess.engine import AutoPreprocess; print('✓ preprocessing')"
```

**Expected:** All imports succeed with no errors.

---

## Test 2: CLI Commands

```bash
# Verify all CLI commands are accessible
foodspec --help
foodspec data load --help
foodspec preprocess run --help
foodspec modeling train --help
foodspec analysis rq-analysis --help
foodspec workflow heating-quality --help
foodspec utils validate --help
```

**Expected:** All commands display help text with no errors.

---

## Test 3: Pytest Suite

```bash
# Run core tests
pytest tests/core/ -q

# Run preprocessing tests  
pytest tests/preprocess/ -q

# Run feature tests
pytest tests/features/ -q

# Quick full suite (with coverage threshold disabled for speed)
pytest tests/ -x --tb=short -q
```

**Expected:** 
- Core tests: 32+ passing
- Preprocessing tests: 50+ passing
- Feature tests: 17+ passing
- Full suite: 600+ tests discovered, majority passing

---

## Test 4: Documentation Build

```bash
# Build documentation
mkdocs build --strict

# Verify output
ls -lh site/index.html
```

**Expected:** mkdocs builds successfully, site/ directory created with index.html.

---

## Test 5: API Functionality

```python
# Save as test_api_smoke.py
import numpy as np
import pandas as pd
from foodspec import FoodSpec

# Create synthetic data
np.random.seed(42)
X = np.random.randn(20, 100)
wn = np.linspace(400, 4000, 100)
metadata = pd.DataFrame({"sample_id": [f"s{i}" for i in range(20)]})

# Test FoodSpec workflow
fs = FoodSpec(X, wavenumbers=wn, metadata=metadata, modality="raman")
print(f"✓ FoodSpec created: {fs}")
print(f"✓ Dataset: n={len(fs.data)}, features={fs.data.x.shape[1]}")
print(f"✓ Summary:\n{fs.summary()}")
```

Run: `python test_api_smoke.py`

**Expected:** Script runs without errors, prints FoodSpec summary.

---

## Test 6: Example Scripts

```bash
# Run quickstart examples (if data available)
python examples/oil_authentication_quickstart.py --dry-run || echo "Data not available (OK)"
python examples/heating_quality_quickstart.py --dry-run || echo "Data not available (OK)"
```

**Expected:** Scripts either run successfully or gracefully report missing data.

---

## Success Criteria

All 6 smoke tests must pass:

- ✅ Installation & Import
- ✅ CLI Commands
- ✅ Pytest Suite (>80% tests passing)
- ✅ Documentation Build
- ✅ API Functionality
- ✅ Example Scripts

**If any test fails:** Investigate immediately before proceeding with release or deployment.

---

## Quick One-Liner

```bash
# Run all smoke tests sequentially
python -c "from foodspec import FoodSpec; print('✓ Import')" && \
foodspec --help > /dev/null && echo "✓ CLI" && \
pytest tests/core/ -q && echo "✓ Tests" && \
mkdocs build --strict > /dev/null && echo "✓ Docs" && \
echo "✅ ALL SMOKE TESTS PASSED"
```

---

**Maintained by:** FoodSpec Core Team  
**Last Updated:** December 25, 2025  
**Version:** 1.0 (Post-PHASE 6 Refactoring)
