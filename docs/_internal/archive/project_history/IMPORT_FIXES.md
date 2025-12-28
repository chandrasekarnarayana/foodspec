# Import Fixes Reference

## Summary of Issues Found (15 total)

### 1. `from foodspec.core import SpectralDataset`
- **Files**: docs/06-developer-guide/documentation_guidelines.md, docs/index.md
- **Fix**: `from foodspec import SpectralDataset`

### 2. `from foodspec.core.spectral_dataset import PreprocessOptions`
- **Files**: examples/hyperspectral_demo.py, examples/spectral_dataset_demo.py
- **Fix**: `from foodspec.core.spectral_dataset import PreprocessingConfig`

### 3. `from foodspec.data import load_library`
- **Files**: docs/01-getting-started/getting_started.md, docs/01-getting-started/quickstart_python.md, docs/04-user-guide/libraries.md, docs/03-cookbook/validation_chemometrics_oils.md
- **Fix**: `from foodspec import load_library`

### 4. `from foodspec.preprocessing.*` (No such module)
- **Files**: docs/02-tutorials/reference_analysis_oil_authentication.md
- **Module is**: `foodspec.preprocess` (not "preprocessing")
- **Fixes**:
  - `from foodspec.preprocessing.baseline_correction import als_baseline_correction` → `from foodspec.preprocess.baseline import baseline_als` OR `from foodspec import baseline_als`
  - `from foodspec.preprocessing.smoothing import savgol_smooth` → Use `from foodspec.preprocess.smoothing import SavitzkyGolaySmoother` OR raw scipy
  - `from foodspec.preprocessing.normalization import normalize_unit_vector` → `from foodspec.preprocess.normalization import VectorNormalizer`

### 5. `from foodspec.harmonization import align_wavenumbers`
- **Files**: docs/03-cookbook/cookbook_preprocessing.md
- **Fix**: Check harmonization module - function may not exist or have different name

### 6. `from foodspec.ml import nested_cross_validate`
- **Files**: docs/08-api/ml.md
- **Fix**: Function may not exist - check ml module

### 7. `from foodspec.preprocess import ALSBaseline`
- **Files**: docs/08-api/index.md
- **Fix**: `from foodspec.preprocess.baseline import ALSBaseline`

### 8. `from foodspec.workflows import OilAuthenticationWorkflow`
- **Files**: docs/06-developer-guide/documentation_guidelines.md, docs/index.md
- **Fix**: This class doesn't exist in workflows - workflow functions are in apps.oils

### 9. Multiline imports (parsing errors)
- `from foodspec.features.rq import (`
- `from foodspec.metrics import (`
- `from foodspec.stats.fusion_metrics import (`
- `from foodspec.viz import (`
- **Issue**: Audit script can't parse multiline imports
- **Action**: These need manual review in their files

## Common Patterns

1. **Top-level exports**: Many things are exported from `foodspec` directly - use those
2. **Module naming**: `preprocess` not "preprocessing", no "validation" submodule
3. **Class names**: `PreprocessingConfig` not "PreprocessOptions"
4. **Workflows**: Functions are in `apps.*`, not classes in `workflows.*`
