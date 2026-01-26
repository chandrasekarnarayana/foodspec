# FoodSpec Preprocessing Engine - Design & Implementation Plan

## Overview

Implement a robust, extensible spectroscopy preprocessing subsystem supporting Raman, FTIR, and shared techniques, exposed via YAML recipes, Python API, and CLI integration.

---

## Architecture Overview

### Module Structure

```
src/foodspec/preprocess/
├── __init__.py                  # Package exports
├── engine.py                    # Core engine: PreprocessRecipe, PreprocessEngine
├── operators.py                 # Individual operators (despiking, baseline, etc)
├── registry.py                  # Operator registry & factory
├── data.py                      # Data loading, validation, normalization
├── cache.py                     # Hash-based caching, manifest generation
├── qc.py                        # Visualization, QC plots
├── loaders.py                   # YAML recipe loader
└── presets/
    ├── default.yaml             # Default preset (all modalities)
    ├── raman.yaml               # Raman-specific preset
    ├── ftir.yaml                # FTIR/IR-specific preset
    └── custom/
        ├── oil_auth.yaml        # Oil authentication example
        └── chips_matrix.yaml    # Chips matrix example

tests/preprocess/
├── conftest.py                  # Fixtures (synthetic data)
├── test_data.py                 # Data model, loading, validation
├── test_operators.py            # Individual operators
├── test_registry.py             # Registry, factory
├── test_engine.py               # Pipeline orchestration
├── test_recipe.py               # Recipe validation, YAML loading
├── test_cache.py                # Caching, manifests
├── test_qc.py                   # Visualizations
└── test_integration.py          # Full preprocessing pipelines

docs/
└── preprocessing.md             # User documentation
```

---

## Data Model (data.py)

### Input Formats

**Format 1: Wide CSV** (columns = wavenumbers)
```
sample_id,batch,instrument,1000,1001,1002,...,3000
oil_1,B1,Raman_1,0.234,0.245,...,0.198
oil_2,B1,Raman_1,0.256,0.267,...,0.210
```

**Format 2: Long CSV** (columns = sample_id, x, y, metadata)
```
sample_id,batch,instrument,wavenumber,intensity
oil_1,B1,Raman_1,1000,0.234
oil_1,B1,Raman_1,1001,0.245
oil_2,B1,Raman_1,1000,0.256
```

### Standard Representation

```python
# Internal representation (always this after normalization)
class SpectraData:
    X: np.ndarray              # (n_samples, n_features)
    wavenumbers: np.ndarray    # (n_features,) sorted wavenumber grid
    metadata: pd.DataFrame     # (n_samples, metadata_cols)
    modality: Literal['raman', 'ftir', 'ir', 'unknown']
```

### Metadata Columns (Optional)

Supported: batch, stage, instrument, replicate, matrix, modality, sample_id

---

## Operator Design (operators.py)

### Base Class

```python
class PreprocessOperator(ABC):
    """Base for all preprocessing operators."""
    
    name: str  # Registry key (e.g., "savgol")
    modality_support: List[str]  # ['raman', 'ftir'] or ['all']
    
    def fit(self, X, meta=None):
        """Optional: learn parameters from data (e.g., baseline)."""
        pass
    
    def transform(self, X, meta=None) -> np.ndarray:
        """Apply operator. X shape: (n_samples, n_features)."""
        pass
    
    def to_dict(self) -> Dict:
        """Serialize for manifest."""
        pass
```

### Operators to Implement (v1)

| Operator | Class | Modality | Purpose |
|----------|-------|----------|---------|
| `savgol` | SavitzkyGolayFilter | all | Smoothing, also enables derivatives |
| `derivative` | DerivativeOperator | all | 1st/2nd derivative (requires prior smoothing) |
| `snv` | StandardNormalVariate | all | Normalization (subtract mean, divide std) |
| `area_norm` | AreaNormalization | all | Normalize by total area |
| `baseline_als` | ALSBaseline | all | Asymmetric least squares baseline removal |
| `despike` | DespikeOperator | raman | Remove cosmic rays (median filter) |
| `fluorescence` | FluorescenceRemoval | raman | Baseline-fitted fluorescence removal |
| `emsc` | EMSCOperator | ftir | Extended multiplicative scatter correction |
| `msc` | MSCOperator | ftir | Multiplicative scatter correction |
| `interpolate` | InterpolationOperator | all | Align to reference grid or resample |

---

## Core Engine (engine.py)

### PreprocessRecipe (Validated Dataclass)

```python
@dataclass
class PreprocessRecipe:
    modality: Literal['raman', 'ftir', 'ir', 'unknown']
    preset: Optional[str] = None  # e.g., "default", "oil_auth"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Serialization/deserialization
    def to_dict(self) -> Dict
    def to_yaml(self) -> str
    @classmethod
    def from_yaml(cls, path) -> PreprocessRecipe
    @classmethod
    def from_dict(cls, d) -> PreprocessRecipe
    
    # Validation
    def validate(self) -> None
```

### PreprocessEngine

```python
class PreprocessEngine:
    """Orchestrate preprocessing pipeline."""
    
    def __init__(self, recipe: PreprocessRecipe, seed=None, cache_dir=None):
        self.recipe = recipe
        self.seed = seed or 42
        self.cache_dir = cache_dir
        self.operators = []  # Instantiated operators in order
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Instantiate operators from recipe."""
        
    def fit(self, X, meta=None):
        """Fit all operators (in-place state)."""
    
    def transform(self, X, meta=None) -> Dict:
        """Apply pipeline. Returns {X_processed, meta, provenance}."""
    
    def fit_transform(self, X, meta=None) -> Dict:
        """Fit and transform."""
    
    def get_provenance(self) -> Dict:
        """Metadata about preprocessing run."""
    
    @classmethod
    def from_recipe_path(cls, recipe_yaml, **kwargs):
        """Load from YAML file."""
```

---

## YAML Schema & Recipe System

### Schema

```yaml
# Global preprocess section
preprocess:
  modality: raman|ftir|ir
  preset: default|oil_auth|chips_matrix  # Optional, merged with steps
  steps:
    - op: savgol
      window: 11
      poly: 3
    - op: baseline_als
      lam: 1e5
      p: 0.01
    - op: snv
```

### Preset Files Location

```
src/foodspec/preprocess/presets/
├── default.yaml
├── raman.yaml
├── ftir.yaml
└── custom/
    ├── oil_auth.yaml
    └── chips_matrix.yaml
```

### Example Preset (raman.yaml)

```yaml
modality: raman
description: "Standard Raman preprocessing: despike, fluorescence, smooth, SNV"
steps:
  - op: despike
    window: 5
  - op: fluorescence
    method: poly
    poly_order: 2
  - op: savgol
    window: 11
    poly: 3
  - op: snv
```

### Loader/Merger (loaders.py)

```python
def load_recipe(protocol_config, preset_name=None, cli_overrides=None) -> PreprocessRecipe:
    """
    1. Load preset if preset_name given
    2. Merge protocol preprocess section
    3. Apply CLI overrides
    4. Validate & return
    """
```

---

## Caching & Provenance (cache.py)

### Hash Key Components

```python
cache_key = hash({
    "raw_data_hash": md5(raw_X + wavenumbers),
    "recipe_hash": md5(recipe.to_yaml()),
    "foodspec_version": "2.0.0",
    "operators_versions": {op: version for op in ops}
})
```

### Preprocess Manifest

Saved as `outdir/data/preprocess_manifest.json`:

```json
{
  "run_id": "exp_20250126_150630",
  "recipe": {
    "modality": "raman",
    "preset": "default",
    "steps": [
      {"op": "savgol", "window": 11, "poly": 3},
      {"op": "baseline_als", "lam": 1e5, "p": 0.01}
    ]
  },
  "cache_key": "abc123...",
  "foodspec_version": "2.0.0",
  "timestamps": {
    "start": "2025-01-26T15:06:30",
    "end": "2025-01-26T15:06:35"
  },
  "statistics": {
    "n_samples_input": 100,
    "n_samples_output": 98,
    "n_features": 1024,
    "rejected_spectra": 2,
    "rejection_reason": ["cosmic_ray", "nan_values"]
  },
  "operators_applied": [
    {"op": "savgol", "time_ms": 12.3},
    {"op": "baseline_als", "time_ms": 45.6}
  ],
  "warnings": ["Spectrum 5 had NaN after baseline removal"]
}
```

---

## QC & Visualization (qc.py)

### QC Plots

1. **Raw vs Processed Overlay**
   - Plot sampled spectra (e.g., 5 random + extremes)
   - X-axis: wavenumber, Y-axis: intensity
   - Two traces per sample (raw, processed)
   - Saved: `figures/raw_vs_processed_overlay.png`

2. **Baseline Estimate Overlay** (if baseline op used)
   - Plot baseline estimates for sampled spectra
   - Saved: `figures/baseline_estimate_overlay.png`

3. **Outlier Detection Summary** (v1: optional)
   - Histogram of spectral norms, mean spectra distance
   - Highlight rejected spectra
   - Saved: `figures/outlier_detection_summary.png`

---

## Protocol Integration

### Step Type Addition

Add new step type `preprocessing` to protocol:

```yaml
protocol:
  name: "Oil_Authentication"
  steps:
    - type: preprocessing
      preset: oil_auth
      override_steps:
        - op: savgol
          window: 7  # Override preset
        
    - type: feature_extraction
      method: pca
      n_components: 10
```

### Code Integration

In `src/foodspec/protocol/steps/__init__.py`:

```python
from foodspec.preprocess import PreprocessStep

STEP_TYPES = {
    'preprocessing': PreprocessStep,
    ...
}
```

Create `src/foodspec/protocol/steps/preprocessing.py`:

```python
class PreprocessStep:
    def execute(self, X, meta, config):
        """Called by ProtocolRunner."""
        engine = PreprocessEngine.from_recipe_dict(config)
        result = engine.fit_transform(X, meta)
        return result['X'], result['meta'], result.get('provenance')
```

---

## Tests (tests/preprocess/)

### Test Coverage

1. **Data Model** (test_data.py)
   - Load wide CSV → standard form
   - Load long CSV → standard form
   - Metadata validation
   - Modality detection

2. **Operators** (test_operators.py)
   - Each operator produces correct shape
   - Deterministic with seed
   - Parameters validated
   - Edge cases (empty, NaN, outliers)

3. **Registry** (test_registry.py)
   - All operators registered
   - Factory creates correct instance
   - Unknown operator raises error

4. **Engine** (test_engine.py)
   - Pipeline executes in order
   - fit/transform/fit_transform work
   - Provenance generated
   - Reproducible with seed

5. **Recipe** (test_recipe.py)
   - YAML parsing
   - Validation
   - Merging presets + overrides
   - Serialization round-trip

6. **Cache** (test_cache.py)
   - Cache hit when data/recipe same
   - Cache miss when recipe changed
   - Manifest written correctly

7. **Integration** (test_integration.py)
   - Full Raman pipeline
   - Full FTIR pipeline
   - Protocol runner integration

---

## Example Protocol Snippet

```yaml
protocol:
  name: "Oil_Authentication_v1"
  modality: raman
  description: "Authenticate edible oils using Raman spectroscopy"
  
  steps:
    - type: preprocessing
      preset: oil_auth
      override_steps:
        - op: despike
          window: 5
        - op: baseline_als
          lam: 1e6
    
    - type: feature_extraction
      method: pca
      n_components: 50
    
    - type: classification
      model: lightgbm
      validation: lobo
```

---

## Files to Create

### Core Implementation

1. `src/foodspec/preprocess/__init__.py` — Package exports
2. `src/foodspec/preprocess/data.py` — Data model, loaders, validators
3. `src/foodspec/preprocess/operators.py` — All 10 operators
4. `src/foodspec/preprocess/registry.py` — Registry & factory
5. `src/foodspec/preprocess/engine.py` — Pipeline engine
6. `src/foodspec/preprocess/loaders.py` — YAML recipe loading
7. `src/foodspec/preprocess/cache.py` — Caching, manifests
8. `src/foodspec/preprocess/qc.py` — QC plots, visualization
9. `src/foodspec/protocol/steps/preprocessing.py` — Protocol integration

### Presets

10. `src/foodspec/preprocess/presets/default.yaml`
11. `src/foodspec/preprocess/presets/raman.yaml`
12. `src/foodspec/preprocess/presets/ftir.yaml`
13. `src/foodspec/preprocess/presets/custom/oil_auth.yaml`
14. `src/foodspec/preprocess/presets/custom/chips_matrix.yaml`

### Tests

15. `tests/preprocess/__init__.py`
16. `tests/preprocess/conftest.py` — Synthetic fixtures
17. `tests/preprocess/test_data.py`
18. `tests/preprocess/test_operators.py`
19. `tests/preprocess/test_registry.py`
20. `tests/preprocess/test_engine.py`
21. `tests/preprocess/test_recipe.py`
22. `tests/preprocess/test_cache.py`
23. `tests/preprocess/test_qc.py`
24. `tests/preprocess/test_integration.py`

### Documentation

25. `docs/preprocessing.md`

---

## Implementation Order

1. **Data model** (data.py) — Foundation
2. **Operators & registry** (operators.py, registry.py) — Core building blocks
3. **Engine** (engine.py) — Orchestration
4. **Recipe loading** (loaders.py) — YAML support
5. **Caching** (cache.py) — Production features
6. **QC** (qc.py) — Visualization
7. **Protocol integration** (preprocessing.py in steps)
8. **Presets** — YAML config files
9. **Tests** — Comprehensive coverage
10. **Documentation** — User guide

---

## Key Design Decisions

1. **Composition over inheritance**: Operators are composable; pipelines are lists, not nested objects.
2. **Immutability during transform**: `transform()` doesn't modify state; `fit()` is explicit.
3. **Metadata-aware**: Operators can use metadata (batch, instrument) for group-aware corrections.
4. **Determinism**: All randomness seeded; same seed = same output.
5. **Caching transparent**: Hash-based, automatic; no manual cache management.
6. **Extensibility**: Adding new operator = 1 class + registry entry + YAML preset.
7. **Minimal deps**: numpy, scipy, sklearn only.

---

## Success Criteria

- [x] Design document complete (this file)
- [ ] Data model (wide/long CSV, normalization)
- [ ] All operators implemented & tested
- [ ] Registry & factory working
- [ ] Engine composable & reproducible
- [ ] Recipe YAML schema + loader
- [ ] Caching system with manifests
- [ ] QC plots generated
- [ ] Protocol integration working
- [ ] 50+ unit tests passing
- [ ] Integration tests passing
- [ ] User documentation complete
- [ ] Example protocols work end-to-end

---

## Next Steps

1. Implement data.py (foundation)
2. Implement operators.py & registry.py (core ops)
3. Implement engine.py (orchestration)
4. Implement loaders.py (YAML support)
5. Implement cache.py & qc.py (production features)
6. Create presets
7. Integrate with protocol system
8. Create tests
9. Create documentation
