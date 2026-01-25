# FoodSpec Migration Guide: v1.x → v2.0.0

## Overview

FoodSpec v2.0.0 introduces a modern, protocol-driven architecture that significantly improves:
- Code organization and maintainability
- Reproducibility and governance
- Testing and documentation
- Performance and extensibility

This guide helps you migrate from v1.x to v2.0.0.

## Timeline

- **v1.1.0** (Current): Deprecation warnings added
- **v1.2.0-1.4.0**: Migration support period
- **v2.0.0**: Deprecated code removed

## Import Changes

### Core Modules

```python
# ❌ Old (deprecated)
from foodspec.spectral_dataset import SpectralDataset

# ✅ New
from foodspec.core import SpectralDataset
```

```python
# ❌ Old (deprecated)
from foodspec.output_bundle import OutputBundle

# ✅ New
from foodspec.core import OutputBundle
```

### Preprocessing

```python
# ❌ Old (deprecated)
from foodspec.preprocessing_pipeline import Pipeline

# ✅ New
from foodspec.preprocess import PreprocessingEngine
```

### Machine Learning

```python
# ❌ Old (deprecated)
from foodspec.model_lifecycle import ModelLifecycle

# ✅ New
from foodspec.ml import ModelLifecycle
```

### Reporting

```python
# ❌ Old (deprecated)
from foodspec.reporting import generate_report

# ✅ New
from foodspec.reporting import generate_dossier
```

### I/O Operations

```python
# ❌ Old (deprecated)
from foodspec.spectral_io import load_spectra

# ✅ New
from foodspec.io import load_folder, read_spectra
```

## API Changes

### FoodSpec Unified API

The new `FoodSpec` class provides a unified entry point:

```python
# ✅ New unified API
from foodspec import FoodSpec

# Initialize
fs = FoodSpec()

# Load data
dataset = fs.load_folder("data/")

# Run analysis
result = fs.run_analysis(dataset, protocol="oil_authentication")

# Generate report
fs.generate_dossier(result, output_dir="results/")
```

### Protocol-Driven Workflows

```python
# ✅ New protocol system
from foodspec.protocol import Protocol

# Define protocol
protocol = Protocol.from_yaml("my_protocol.yaml")

# Run protocol
result = protocol.run(data)
```

## Common Migration Patterns

### Pattern 1: Basic Analysis

**Old Code:**
```python
from foodspec import FoodSpectrumSet
from foodspec.preprocessing_pipeline import Pipeline

# Load data
data = FoodSpectrumSet.from_folder("data/")

# Preprocess
pipeline = Pipeline()
pipeline.add_step("baseline", method="als")
preprocessed = pipeline.fit_transform(data)
```

**New Code:**
```python
from foodspec import FoodSpec
from foodspec.preprocess import PreprocessingEngine

# Load data
fs = FoodSpec()
data = fs.load_folder("data/")

# Preprocess
engine = PreprocessingEngine()
engine.add_step("baseline", method="als")
preprocessed = engine.fit_transform(data)
```

### Pattern 2: Model Training

**Old Code:**
```python
from foodspec.model_lifecycle import train_model

model = train_model(X, y, algorithm="rf")
```

**New Code:**
```python
from foodspec.ml import ModelLifecycle

lifecycle = ModelLifecycle()
model = lifecycle.train(X, y, algorithm="rf")
```

### Pattern 3: Report Generation

**Old Code:**
```python
from foodspec.reporting import generate_report

generate_report(results, output_path="report.html")
```

**New Code:**
```python
from foodspec.reporting import generate_dossier

generate_dossier(results, output_dir="results/")
```

## Automated Migration

Use the migration checker tool:

```bash
# Check for deprecated usage
foodspec-check-migration /path/to/your/code

# Apply automatic fixes
foodspec-migrate --apply /path/to/your/code
```

## Troubleshooting

### Issue: DeprecationWarning flooding output

**Solution:** Suppress warnings temporarily (not recommended for production):

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

Better: Fix the deprecated usage.

### Issue: Import error after migration

**Solution:** Check the import mapping table in this guide.

### Issue: Functionality seems missing

**Solution:** Some features were reorganized. Check the API documentation.

## Getting Help

- GitHub Issues: https://github.com/chandrasekarnarayana/foodspec/issues
- Documentation: https://foodspec.readthedocs.io
- Migration FAQ: docs/migration/faq.md

## Complete Import Mapping

| Old Import | New Import | Status |
|-----------|------------|--------|
| `foodspec.spectral_dataset` | `foodspec.core` | Deprecated v1.1.0, Remove v2.0.0 |
| `foodspec.output_bundle` | `foodspec.core` | Deprecated v1.1.0, Remove v2.0.0 |
| `foodspec.model_lifecycle` | `foodspec.ml` | Deprecated v1.1.0, Remove v2.0.0 |
| `foodspec.preprocessing_pipeline` | `foodspec.preprocess` | Deprecated v1.1.0, Remove v2.0.0 |
| `foodspec.spectral_io` | `foodspec.io` | Deprecated v1.1.0, Remove v2.0.0 |
| `foodspec.reporting` | `foodspec.reporting` (package) | Deprecated v1.1.0, Remove v2.0.0 |
| ... | ... | ... |

## Timeline Summary

### v1.1.0 (Current)
- ⚠️  Deprecation warnings added
- ✅ All old code still works
- 📖 Migration guide published

### v1.2.0-1.4.0 (Months 1-4)
- 🔨 Migration support
- 🐛 Bug fixes
- 📚 Documentation updates

### v2.0.0 (Month 6)
- 🗑️  Deprecated code removed
- ✨ Clean, modern API
- 🚀 Performance improvements

**Recommendation:** Start migrating now to avoid last-minute issues.
