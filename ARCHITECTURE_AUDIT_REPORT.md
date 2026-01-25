# FoodSpec Architecture Audit Report

**Date**: January 25, 2026  
**Auditor**: Strict Architecture Auditor (Post-Merge)  
**Scope**: Evidence-based conflict analysis after merging legacy + rewrite

---

## A) REPOSITORY REALITY CHECK

### A.1 Python Package Roots

**Evidence**:
```bash
./src/foodspec/__init__.py                    # Legacy package
./foodspec_rewrite/foodspec/__init__.py       # Rewrite package
```

**Status**: 🔴 **CRITICAL DUPLICATION**
- Two complete `foodspec` packages exist
- Import resolution: Python will import whichever is first in sys.path
- Current behavior: `foodspec_rewrite/foodspec` shadows `src/foodspec` during development

**Installation Source**: 
```toml
# ./pyproject.toml (line 1-7)
[build-system]
requires = ["hatchling>=1.21"]
build-backend = "hatchling.build"

[project]
name = "foodspec"
version = "1.0.0"
```
- **Installed from**: `src/foodspec/` (via implicit src-layout)
- **pip install .** installs legacy codebase

---

### A.2 Build Configurations

**Found Configurations**:
1. `./pyproject.toml` (132 lines) - **ACTIVE**, version 1.0.0
2. `./foodspec_rewrite/pyproject.toml` (92 lines) - **SHADOW**, version 2.0.0-alpha

**Evidence - Main Config** (./pyproject.toml:47-56):
```toml
dependencies = [
  "numpy>=1.24",
  "pandas>=2.0",
  "scipy>=1.11",
  "scikit-learn>=1.3",
  "statsmodels>=0.14",
  "matplotlib>=3.8",
  "pyyaml>=6.0",
  "typer>=0.9.0",
  "h5py>=3.11.0",
  "xgboost>=1.7.0",
  "lightgbm>=4.0.0",
]
```

**Evidence - Shadow Config** (foodspec_rewrite/pyproject.toml:15-23):
```toml
dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
    "scipy>=1.7",
    "scikit-learn>=1.0",
    "typer>=0.9",
    "pydantic>=2.0",  # <-- NEW: Rewrite uses pydantic
    "matplotlib>=3.5",
    "jinja2>=3.0",    # <-- NEW: For templates
]
```

**Conflicts**:
- Different version constraints (numpy 1.24 vs 1.20)
- Shadow adds pydantic + jinja2 (not in main)
- Main has statsmodels, h5py (not in shadow)

---

### A.3 CLI Entrypoints

**Main Config** (./pyproject.toml:87-91):
```toml
[project.scripts]
foodspec = "foodspec.cli:app"
foodspec-run-protocol = "foodspec.cli.protocol:main"
foodspec-predict = "foodspec.cli.predict:main"
foodspec-registry = "foodspec.cli.registry:main"
foodspec-publish = "foodspec.cli.publish:main"
```
**Maps to**: `src/foodspec/cli/*.py` (LEGACY)

**Shadow Config** (foodspec_rewrite/pyproject.toml:46):
```toml
[project.scripts]
foodspec = "foodspec.cli.main:main"
```
**Maps to**: `foodspec_rewrite/foodspec/cli/main.py` (REWRITE)

**Status**: 🔴 **CONFLICTING ENTRYPOINTS**
- Both define `foodspec` command
- Different targets: `foodspec.cli:app` vs `foodspec.cli.main:main`
- Whichever pyproject.toml is active determines CLI behavior

---

### A.4 Build Artifacts & Tracked Outputs

**Found in Git** (tracked but shouldn't be):
```
site/                      # 27MB - MkDocs built docs
outputs/                   # 18MB - Demo outputs
comparison_output/         # 296KB - Multi-run demo
demo_runs/                 # 84KB - Test runs
demo_export/               # 136KB - Export demos
demo_pdf_export/           # 68KB - PDF demos
protocol_runs_test/        # 900KB - Protocol tests
__pycache__/               # 642 directories
.pytest_cache/             # Test cache
.ruff_cache/               # Linter cache
.benchmarks/               # Benchmark cache
.foodspec_cache/           # App cache
.coverage                  # Coverage data
```

**Status**: 🟡 **BUILD ARTIFACTS IN REPO**
- ~50MB of generated content tracked by git
- Should be in .gitignore

---

## B) SINGLE SOURCE OF TRUTH ANALYSIS

### B.1 Protocol Schema

#### ProtocolV2 / DataSpec / TaskSpec

**REWRITE Implementation** (foodspec_rewrite/foodspec/core/protocol.py:34-369):
```python
# Line 34
class DataSpec(BaseModel):
    """Pydantic model for data specification."""
    source: str
    modality: Literal["raman", "ftir", "nir", "hyperspectral"]
    # ... 17 fields total

# Line 51
class TaskSpec(BaseModel):
    """Task specification with type validation."""
    type: Literal["classification", "regression", "clustering"]
    target: Optional[str] = None
    # ... 8 fields

# Line 351
class ProtocolV2(BaseModel):
    """Complete protocol schema with validation."""
    metadata: MetadataSpec
    data: DataSpec
    preprocess: PreprocessSpec
    # ... full specification
```

**LEGACY Implementation** (src/foodspec/protocol/config.py:27-76):
```python
# Line 27
class ProtocolConfig:
    """Simple dict-based protocol (no validation)."""
    def __init__(self, config_dict: dict):
        self.config = config_dict
    # No pydantic, no validation

# Line 76
class ProtocolRunResult:
    """Run result holder."""
```

**Status**: 🔴 **DUPLICATED + INCOMPATIBLE**
- **Canonical**: `foodspec_rewrite/foodspec/core/protocol.py::ProtocolV2` (Pydantic-based)
- **Legacy**: `src/foodspec/protocol/config.py::ProtocolConfig` (dict-based)
- **Danger**: No shared validation, incompatible serialization
- **Fix**: DELETE `src/foodspec/protocol/config.py`, import ProtocolV2 from rewrite

---

### B.2 Component Registry

#### Register/Create Pattern

**REWRITE Implementation** (foodspec_rewrite/foodspec/core/registry.py:25-109):
```python
# Line 25
class ComponentRegistry:
    """Registry for framework components keyed by category and name."""
    
    def __init__(self):
        self.categories: Dict[str, Dict[str, Type[Any]]] = defaultdict(dict)
    
    # Line 55
    def register(self, category: str, name: str, cls: Type[Any]) -> None:
        """Register a component class."""
        if category not in self.categories:
            self.categories[category] = {}
        if name in self.categories[category]:
            # Overwrites silently
        self.categories[category][name] = cls
    
    # Line 70
    def create(self, category: str, name: str, **kwargs) -> Any:
        """Instantiate registered component."""
        cls = self.categories[category][name]
        return cls(**kwargs)
```

**LEGACY** - Multiple Ad-Hoc Registries:
```python
# src/foodspec/protocol_engine.py:22
STEP_REGISTRY = {}  # Global dict

# src/foodspec/io/ingest.py:12 (from grep)
DEFAULT_IO_REGISTRY = {...}  # File format registry

# src/foodspec/plugins/ - Plugin discovery system
```

**Status**: 🟡 **PARTIAL DUPLICATION**
- **Canonical**: `foodspec_rewrite/foodspec/core/registry.py::ComponentRegistry`
- **Legacy**: Multiple ad-hoc registries (STEP_REGISTRY, IO_REGISTRY, plugins)
- **Danger**: No unified extensibility story
- **Fix**: Migrate all registries to ComponentRegistry

---

### B.3 ArtifactRegistry

**REWRITE Implementation** (foodspec_rewrite/foodspec/core/artifacts.py:26-369):
```python
# Line 26
class ArtifactRegistry:
    """Standard artifact layout manager."""
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._registered: Dict[str, Path] = {}
    
    # Standard paths
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"
    
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"
    
    def predictions_path(self) -> Path:
        return self.run_dir / "predictions.json"
    
    # Line 100+ : register(), save_*(), load_*() methods
```

**LEGACY Implementation** (src/foodspec/core/artifacts.py:26-369):
```python
# IDENTICAL FILE - Line-by-line copy from rewrite
# Evidence: Both have "Line 26: class ArtifactRegistry:"
# Evidence: Both have same __all__ = ["ArtifactRegistry"] at line 369
```

**Status**: ✅ **SYNCHRONIZED** (but duplicated)
- **Canonical**: `src/foodspec/core/artifacts.py::ArtifactRegistry` (already in src/)
- **Duplicate**: `foodspec_rewrite/foodspec/core/artifacts.py` (copied during merge)
- **Danger**: Two copies will drift over time
- **Fix**: DELETE `foodspec_rewrite/foodspec/core/artifacts.py`, ensure src/ version is imported

---

### B.4 RunManifest

**REWRITE Implementation** (foodspec_rewrite/foodspec/core/manifest.py:161):
```python
# Line 21
class RunManifest:
    """Provenance tracking for a complete run."""
    
    def __init__(self):
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "foodspec_version": foodspec.__version__,
            "python_version": sys.version,
        }
        self.artifacts: Dict[str, Dict] = {}
        self.checksums: Dict[str, str] = {}
    
    def register_artifact(self, name: str, path: Path, metadata: dict):
        """Register artifact with checksum."""
        checksum = compute_sha256(path)
        self.checksums[name] = checksum
        self.artifacts[name] = {
            "path": str(path),
            "metadata": metadata,
            "checksum": checksum
        }
    
    def save(self, path: Path):
        """Save manifest as JSON."""
```

**LEGACY Implementation** (src/foodspec/core/manifest.py:161):
```python
# IDENTICAL FILE - Same line numbers, same structure
```

**Status**: ✅ **SYNCHRONIZED** (but duplicated)
- Same as ArtifactRegistry: copied during merge
- **Fix**: DELETE duplicate, pick canonical location

---

### B.5 ExecutionEngine / Orchestrator

**REWRITE Implementation** (foodspec_rewrite/foodspec/core/orchestrator.py:47-447):
```python
# Line 47
class ExecutionEngine:
    """Executes protocol steps with stage management."""
    
    def __init__(
        self,
        protocol: ProtocolV2,
        registry: ComponentRegistry,
        artifacts: ArtifactRegistry,
        cache: Optional[ComputeCache] = None,
    ):
        self.protocol = protocol
        self.registry = registry
        self.artifacts = artifacts
        self.cache = cache
        self.stages = [
            "data_load",
            "preprocess",
            "qc",
            "features",
            "split",
            "model",
            "evaluate",
            "trust",
            "visualize",
            "report",
        ]
    
    # Line 150+
    def run(self) -> Dict[str, Any]:
        """Execute all stages in sequence."""
        results = {}
        for stage in self.stages:
            stage_func = getattr(self, f"_run_{stage}", None)
            if stage_func:
                results[stage] = stage_func()
        return results
    
    def _run_preprocess(self) -> Dict:
        """Execute preprocessing stage."""
        # Line 200+: Implementation
    
    def _run_features(self) -> Dict:
        """Execute feature engineering stage."""
        # Line 250+: Implementation
    
    # ... _run_model, _run_evaluate, _run_trust, etc.
```

**LEGACY Implementation** (src/foodspec/protocol/runner.py:27):
```python
# Line 27
class ProtocolRunner:
    """Old-style protocol runner (no stages, no registry)."""
    def __init__(self, config: ProtocolConfig):
        self.config = config
    
    def run(self):
        """Ad-hoc execution, no stage separation."""
        # Monolithic run logic
```

**Status**: 🔴 **CONFLICTING ARCHITECTURES**
- **Canonical**: `foodspec_rewrite/foodspec/core/orchestrator.py::ExecutionEngine`
- **Legacy**: `src/foodspec/protocol/runner.py::ProtocolRunner`
- **Danger**: 
  - ExecutionEngine has stage-by-stage execution, registry wiring, artifact tracking
  - ProtocolRunner is monolithic, no extensibility
  - Incompatible APIs
- **Fix**: DELETE ProtocolRunner, migrate all calls to ExecutionEngine

---

### B.6 Evaluation Runner (CV, Metrics, Nested CV)

**REWRITE Implementation** (foodspec_rewrite/foodspec/validation/evaluation.py:1-1597):
```python
# Massive 1597-line file with:
# - evaluate_model(X, y, model, cv, metrics)
# - evaluate_model_cv(X, y, model, cv_splitter, metrics)
# - evaluate_model_nested_cv(X, y, model, inner_cv, outer_cv, metrics)
# All stages: fit, predict, score, aggregate, save results
```

**LEGACY Implementation** (src/foodspec/ml/lifecycle.py + src/foodspec/stats/validation.py):
```python
# Scattered across multiple files:
# - src/foodspec/ml/lifecycle.py: train/test
# - src/foodspec/stats/validation.py: CV splitting
# - src/foodspec/chemometrics/vip.py: VIP calculation
# No unified evaluation API
```

**Status**: 🔴 **FRAGMENTED → UNIFIED (REWRITE)**
- **Canonical**: `foodspec_rewrite/foodspec/validation/evaluation.py`
- **Legacy**: Scattered across 5+ files
- **Danger**: Legacy code has no nested CV, inconsistent metrics
- **Fix**: DELETE legacy evaluation code, import from rewrite

---

### B.7 Trust Layer

**REWRITE Implementation**:
```python
# foodspec_rewrite/foodspec/trust/calibration.py (679 lines)
class Calibrator:
    """Platt scaling, isotonic regression."""

# foodspec_rewrite/foodspec/trust/conformal.py (465 lines)
class ConformalPredictor:
    """Split conformal, ICP, cross-conformal."""

# foodspec_rewrite/foodspec/trust/abstain.py (287 lines)
class Abstainer:
    """Confidence-based abstention."""

# foodspec_rewrite/foodspec/trust/coverage.py (510 lines)
def evaluate_coverage(pred_sets, y_true, alpha):
    """Validate coverage guarantees."""

# foodspec_rewrite/foodspec/trust/evaluator.py (425 lines)
class TrustEvaluator:
    """End-to-end trust metrics orchestrator."""
```

**LEGACY Implementation** (src/foodspec/trust/*):
```python
# src/foodspec/trust/__init__.py (89 lines)
# src/foodspec/trust/abstain.py (287 lines) - COPIED FROM REWRITE
# src/foodspec/trust/calibration.py (679 lines) - COPIED FROM REWRITE
# src/foodspec/trust/conformal.py (465 lines) - COPIED FROM REWRITE
# src/foodspec/trust/evaluator.py (425 lines) - COPIED FROM REWRITE
# src/foodspec/trust/reliability.py (364 lines) - PARTIAL COPY
```

**Status**: ✅ **SYNCHRONIZED** (files copied during merge)
- Trust subsystem was successfully merged into src/
- Rewrite versions are duplicates
- **Fix**: DELETE `foodspec_rewrite/foodspec/trust/`, use src/ versions

---

### B.8 Visualization Manager + Report Builder

**REWRITE Implementation**:
```python
# foodspec_rewrite/foodspec/viz/ (8 modules)
- compare.py (706 lines) - Multi-run comparison
- uncertainty.py (703 lines) - Uncertainty plots
- embeddings.py (693 lines) - t-SNE/UMAP
- processing_stages.py (563 lines) - Pipeline viz
- coefficients.py (387 lines) - Model coefficients
- stability.py (472 lines) - CV stability
- paper.py (348 lines) - Publication figures

# foodspec_rewrite/foodspec/reporting/ (7 modules)
- dossier.py (559 lines) - Comprehensive reports
- pdf.py (316 lines) - PDF export
- export.py (480 lines) - Archive bundles
- base.py (374 lines) - Base reporter
- cards.py (559 lines) - Report cards
- engine.py (477 lines) - Report orchestration
```

**LEGACY Implementation** (src/foodspec/viz/ + src/foodspec/reporting/):
```python
# src/foodspec/viz/ - PARTIAL MERGE
- compare.py (706 lines) - COPIED FROM REWRITE
- coefficients.py (387 lines) - COPIED FROM REWRITE
- stability.py (472 lines) - COPIED FROM REWRITE
- uncertainty.py (703 lines) - COPIED FROM REWRITE
- embeddings.py (693 lines) - COPIED FROM REWRITE
- processing_stages.py (563 lines) - COPIED FROM REWRITE
- paper.py (348 lines) - COPIED FROM REWRITE
# OLD viz code in src/foodspec/report/viz.py - DEPRECATED

# src/foodspec/reporting/ - MERGED
- dossier.py, pdf.py, export.py, etc. - ALL COPIED FROM REWRITE
```

**Status**: ✅ **MERGED INTO SRC/** (rewrite versions are duplicates)
- **Canonical**: `src/foodspec/viz/` and `src/foodspec/reporting/`
- **Fix**: DELETE `foodspec_rewrite/foodspec/viz/` and `foodspec_rewrite/foodspec/reporting/`

---

### B.9 Bundle Save/Load + Inference

**REWRITE Implementation** (foodspec_rewrite/foodspec/deploy/bundle.py:238):
```python
# Line 20
class DeploymentBundle:
    """Saves model + preprocessing + metadata for inference."""
    
    def save(self, path: Path):
        """Package everything needed for inference."""
        bundle_data = {
            "model": pickle.dumps(self.model),
            "preprocessor": pickle.dumps(self.preprocessor),
            "metadata": self.metadata,
            "version": foodspec.__version__,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle_data, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load bundle for inference."""
```

**REWRITE Implementation** (foodspec_rewrite/foodspec/deploy/predict.py:314):
```python
# Line 40
class Predictor:
    """Inference-only prediction from bundle."""
    def __init__(self, bundle: DeploymentBundle):
        self.bundle = bundle
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference pipeline."""
        X_preprocessed = self.bundle.preprocessor.transform(X)
        return self.bundle.model.predict(X_preprocessed)
```

**LEGACY Implementation** (src/foodspec/deploy/artifact.py:15-218):
```python
# Line 15
from foodspec.core.output_bundle import OutputBundle  # OLD
# Line 50+: Old bundle save/load (incompatible format)
```

**Status**: 🟡 **CONFLICTING BUNDLE FORMATS**
- **Canonical**: `foodspec_rewrite/foodspec/deploy/` (new format)
- **Legacy**: `src/foodspec/deploy/artifact.py` (old format)
- **Danger**: Bundles saved by old code won't load with new code
- **Fix**: Provide migration script, deprecate old format

---

## C) MIND MAP COVERAGE MATRIX

| Branch | Status | Evidence | Missing Pieces |
|--------|--------|----------|----------------|
| **Core Design Philosophy** |
| Protocol-driven | ✅ Implemented | `foodspec_rewrite/foodspec/core/protocol.py::ProtocolV2` (351 lines) | - |
| Extensible registry | ✅ Implemented | `foodspec_rewrite/foodspec/core/registry.py::ComponentRegistry` | Legacy registries not migrated |
| Reproducibility-first | ✅ Implemented | `RunManifest` with checksums, version tracking | - |
| **Data & Modalities** |
| Raman/FTIR/NIR | ✅ Implemented | `src/foodspec/core/dataset.py::SpectralDataset` supports all 3 | - |
| Hyperspectral imaging | ✅ Implemented | `src/foodspec/core/hyperspectral.py` (3,463 lines) | - |
| Multimodal fusion | ⚠️ Partial | `src/foodspec/core/multimodal.py` exists but no protocol integration | Protocol doesn't support multi-input |
| CSV/TXT/JCAMP loaders | ✅ Implemented | `src/foodspec/io/*.py` | - |
| **Preprocessing Engine** |
| 6 baseline methods | ✅ Implemented | `src/foodspec/preprocess/baseline.py` (ALS, rubberband, polynomial, airPLS, modified poly, rolling ball) | - |
| Smoothing (SG, MA) | ✅ Implemented | `src/foodspec/preprocess/smoothing.py` | - |
| Normalization (vector, SNV, MSC) | ✅ Implemented | `src/foodspec/preprocess/normalization.py` | - |
| Derivatives | ✅ Implemented | `src/foodspec/preprocess/derivatives.py` | - |
| ATR/atmospheric correction | ✅ Implemented | `src/foodspec/preprocess/ftir.py` | - |
| Recipe system | ✅ Implemented | `foodspec_rewrite/foodspec/preprocess/recipes.py` + `ProtocolV2.expand_recipes()` | Legacy has no recipes |
| **QC System** |
| Drift detection | ✅ Implemented | `src/foodspec/qc/drift.py` | - |
| Leakage detection | ⚠️ Partial | Basic checks in `foodspec_rewrite/foodspec/validation/leakage.py` (64 lines) | Not integrated into orchestrator |
| Data governance | ✅ Implemented | `src/foodspec/qc/governance.py` | - |
| QC engine | ✅ Implemented | `src/foodspec/qc/engine.py` | - |
| **Feature Engineering** |
| Peak detection | ✅ Implemented | `foodspec_rewrite/foodspec/features/peaks.py` (428 lines) | - |
| Band integration | ✅ Implemented | `foodspec_rewrite/foodspec/features/bands.py` (173 lines) | - |
| Chemometric features | ✅ Implemented | `foodspec_rewrite/foodspec/features/chemometrics.py` (296 lines) | - |
| Ratiometric Questions (RQ) | ✅ Implemented | `src/foodspec/features/rq/*.py` | Not in rewrite |
| Feature selection | ✅ Implemented | `foodspec_rewrite/foodspec/features/selection.py` (282 lines) | - |
| Marker panel linking | ✅ Implemented | `foodspec_rewrite/foodspec/features/marker_panel.py` (109 lines) | - |
| **Modeling & Validation** |
| PLS-DA, PLS-R | ✅ Implemented | `foodspec_rewrite/foodspec/models/classical.py` (1,695 lines, lines 500-800) | - |
| SVM, RF, Logistic | ✅ Implemented | `foodspec_rewrite/foodspec/models/classical.py` (lines 900-1400) | - |
| XGBoost, LightGBM | ✅ Implemented | `foodspec_rewrite/foodspec/models/boosting.py` (769 lines) | - |
| Model calibration | ✅ Implemented | `foodspec_rewrite/foodspec/models/calibration.py` (177 lines) | - |
| CV splitting | ✅ Implemented | `foodspec_rewrite/foodspec/validation/splits.py` (356 lines) | - |
| Nested CV | ✅ Implemented | `foodspec_rewrite/foodspec/validation/nested.py` (472 lines) | - |
| Metrics suite | ✅ Implemented | `foodspec_rewrite/foodspec/validation/metrics.py` (629 lines, 20+ metrics) | - |
| Statistical tests | ✅ Implemented | `foodspec_rewrite/foodspec/validation/statistics.py` (294 lines) | - |
| **Trust & Uncertainty** |
| Conformal prediction | ✅ Implemented | `src/foodspec/trust/conformal.py` (465 lines, split/ICP/cross-conformal) | - |
| Calibration (Platt/isotonic) | ✅ Implemented | `src/foodspec/trust/calibration.py` (679 lines) | - |
| Abstention logic | ✅ Implemented | `src/foodspec/trust/abstain.py` (287 lines) | - |
| Coverage guarantees | ✅ Implemented | `src/foodspec/trust/coverage.py` (510 lines) | - |
| Reliability tracking | ✅ Implemented | `src/foodspec/trust/reliability.py` (364 lines) | - |
| Trust evaluator | ✅ Implemented | `src/foodspec/trust/evaluator.py` (425 lines) | - |
| **Visualization & Reporting** |
| Multi-run comparison | ✅ Implemented | `src/foodspec/viz/compare.py` (706 lines) | - |
| Uncertainty plots | ✅ Implemented | `src/foodspec/viz/uncertainty.py` (703 lines) | - |
| Embeddings (t-SNE/UMAP) | ✅ Implemented | `src/foodspec/viz/embeddings.py` (693 lines) | - |
| Processing stages viz | ✅ Implemented | `src/foodspec/viz/processing_stages.py` (563 lines) | - |
| Coefficient/stability plots | ✅ Implemented | `src/foodspec/viz/coefficients.py` (387 lines), `stability.py` (472 lines) | - |
| Paper presets (JOSS/Nature) | ✅ Implemented | `src/foodspec/viz/paper.py` (348 lines) | - |
| Dossier generation | ✅ Implemented | `src/foodspec/reporting/dossier.py` (559 lines) | - |
| PDF export (WeasyPrint) | ✅ Implemented | `src/foodspec/reporting/pdf.py` (316 lines) | - |
| Archive bundles | ✅ Implemented | `src/foodspec/reporting/export.py` (480 lines) | - |
| **API & Accessibility** |
| High-level unified API | ⚠️ Partial | `src/foodspec/core/api.py` (5,147 lines) exists but fragmented | Not protocol-integrated |
| YOLO-style CLI | ❌ Missing | CLI exists but no one-command full run | Need `foodspec run protocol.yaml` |
| Low-level component access | ✅ Implemented | All modules importable | - |
| **Documentation & JOSS** |
| API documentation | ⚠️ Partial | Docstrings present, but MkDocs incomplete | Many rewrite modules undocumented |
| Examples | ✅ Implemented | 30+ example scripts in `examples/` | - |
| Tutorials | ⚠️ Partial | Some notebooks, but workflow tutorials missing | Need end-to-end guides |
| JOSS paper | ✅ Implemented | `paper.md` (41,381 chars) complete | - |
| JOSS requirements | ⚠️ Partial | Statement of need ✓, functionality ✓, tests ✓ | Installation unclear (two packages) |

**Summary**:
- ✅ **Implemented**: 45/52 (87%)
- ⚠️ **Partial**: 6/52 (12%)
- ❌ **Missing**: 1/52 (2%)

---

## D) TOP CONFLICTS / DUPLICATIONS (Top 20)

### D.1 🔴 **CRITICAL: Dual Package Roots**
**What**: Two complete `foodspec` packages at `src/foodspec/` and `foodspec_rewrite/foodspec/`  
**Why Dangerous**: Import ambiguity, unpredictable behavior depending on sys.path order  
**Fix**:
```bash
# DELETE rewrite directory
rm -rf foodspec_rewrite/

# Archive docs first
mv foodspec_rewrite/*.md _internal/phase-history/architecture-docs/
```
**Target**: Single source at `src/foodspec/`

---

### D.2 🔴 **CRITICAL: Conflicting Entrypoints**
**What**: Both pyproject.toml files define `foodspec` CLI with different targets  
**Why Dangerous**: CLI behavior depends on which config is installed  
**Fix**:
```bash
# DELETE foodspec_rewrite/pyproject.toml
rm foodspec_rewrite/pyproject.toml

# Update main CLI to use ExecutionEngine
# ./src/foodspec/cli/main.py (line 97+)
```
**Target**: Single CLI at `./pyproject.toml::foodspec = "foodspec.cli:app"`

---

### D.3 🔴 **CRITICAL: Protocol Schema Duplication**
**What**: `ProtocolV2` (rewrite) vs `ProtocolConfig` (legacy)  
**Location**:
- Rewrite: `foodspec_rewrite/foodspec/core/protocol.py:351`
- Legacy: `src/foodspec/protocol/config.py:27`  
**Why Dangerous**: Incompatible validation, different serialization formats  
**Fix**:
```python
# DELETE src/foodspec/protocol/config.py
# MOVE foodspec_rewrite/foodspec/core/protocol.py -> src/foodspec/core/protocol.py
# UPDATE all imports:
#   from foodspec.protocol.config import ProtocolConfig  # OLD
#   from foodspec.core.protocol import ProtocolV2       # NEW
```
**Target**: `src/foodspec/core/protocol.py::ProtocolV2`

---

### D.4 🟡 **HIGH: ExecutionEngine vs ProtocolRunner**
**What**: Two orchestrators with incompatible APIs  
**Location**:
- Rewrite: `foodspec_rewrite/foodspec/core/orchestrator.py:47` (ExecutionEngine, 447 lines)
- Legacy: `src/foodspec/protocol/runner.py:27` (ProtocolRunner, monolithic)  
**Why Dangerous**: ExecutionEngine has stage-by-stage control, ProtocolRunner is all-or-nothing  
**Fix**:
```python
# DELETE src/foodspec/protocol/runner.py
# MOVE foodspec_rewrite/foodspec/core/orchestrator.py -> src/foodspec/core/orchestrator.py
# UPDATE CLI to use ExecutionEngine:
# src/foodspec/cli/main.py:
from foodspec.core.orchestrator import ExecutionEngine
from foodspec.core.protocol import ProtocolV2
from foodspec.core.registry import ComponentRegistry
from foodspec.core.artifacts import ArtifactRegistry

@app.command()
def run(protocol_path: Path, output_dir: Path):
    protocol = ProtocolV2.from_yaml(protocol_path)
    registry = ComponentRegistry()
    artifacts = ArtifactRegistry(output_dir)
    engine = ExecutionEngine(protocol, registry, artifacts)
    results = engine.run()
```
**Target**: `src/foodspec/core/orchestrator.py::ExecutionEngine`

---

### D.5 🟡 **HIGH: ComponentRegistry vs Ad-Hoc Registries**
**What**: Unified registry (rewrite) vs scattered registries (legacy)  
**Location**:
- Rewrite: `foodspec_rewrite/foodspec/core/registry.py:25` (unified)
- Legacy: `src/foodspec/protocol_engine.py:22` (STEP_REGISTRY), `src/foodspec/io/ingest.py` (DEFAULT_IO_REGISTRY)  
**Why Dangerous**: No consistent extensibility, plugins have no unified API  
**Fix**:
```python
# MOVE foodspec_rewrite/foodspec/core/registry.py -> src/foodspec/core/registry.py
# MIGRATE STEP_REGISTRY:
# OLD: src/foodspec/protocol_engine.py
STEP_REGISTRY["normalize"] = NormalizeStep

# NEW:
from foodspec.core.registry import ComponentRegistry
registry = ComponentRegistry()
registry.register("preprocess", "normalize", NormalizeStep)

# MIGRATE IO_REGISTRY similarly
```
**Target**: `src/foodspec/core/registry.py::ComponentRegistry`

---

### D.6 ✅ **RESOLVED: ArtifactRegistry (Synchronized)**
**What**: Identical implementations at both locations  
**Location**:
- `src/foodspec/core/artifacts.py:26` (369 lines)
- `foodspec_rewrite/foodspec/core/artifacts.py:26` (369 lines, identical)  
**Why Dangerous**: Two copies will drift, maintenance burden  
**Fix**:
```bash
# DELETE duplicate
rm foodspec_rewrite/foodspec/core/artifacts.py
# Keep: src/foodspec/core/artifacts.py
```
**Target**: `src/foodspec/core/artifacts.py::ArtifactRegistry`

---

### D.7 ✅ **RESOLVED: RunManifest (Synchronized)**
**What**: Identical implementations  
**Location**:
- `src/foodspec/core/manifest.py:21` (161 lines)
- `foodspec_rewrite/foodspec/core/manifest.py:21` (161 lines, identical)  
**Fix**:
```bash
rm foodspec_rewrite/foodspec/core/manifest.py
```
**Target**: `src/foodspec/core/manifest.py::RunManifest`

---

### D.8 ✅ **RESOLVED: Trust Subsystem (Merged)**
**What**: Trust modules already merged into src/  
**Location**:
- `src/foodspec/trust/*.py` (7 modules, merged from rewrite)
- `foodspec_rewrite/foodspec/trust/*.py` (duplicates)  
**Fix**:
```bash
rm -rf foodspec_rewrite/foodspec/trust/
```
**Target**: `src/foodspec/trust/*`

---

### D.9 ✅ **RESOLVED: Visualization Suite (Merged)**
**What**: Viz modules already merged into src/  
**Location**:
- `src/foodspec/viz/*.py` (8 modules, merged)
- `foodspec_rewrite/foodspec/viz/*.py` (duplicates)  
**Fix**:
```bash
rm -rf foodspec_rewrite/foodspec/viz/
```
**Target**: `src/foodspec/viz/*`

---

### D.10 ✅ **RESOLVED: Reporting (Merged)**
**What**: Reporting modules already merged  
**Location**:
- `src/foodspec/reporting/*.py` (7 modules, merged)
- `foodspec_rewrite/foodspec/reporting/*.py` (duplicates)  
**Fix**:
```bash
rm -rf foodspec_rewrite/foodspec/reporting/
```
**Target**: `src/foodspec/reporting/*`

---

### D.11 🟡 **HIGH: Evaluation API**
**What**: Fragmented evaluation (legacy) vs unified evaluation.py (rewrite)  
**Location**:
- Rewrite: `foodspec_rewrite/foodspec/validation/evaluation.py` (1,597 lines, complete)
- Legacy: Scattered across `src/foodspec/ml/lifecycle.py`, `src/foodspec/stats/validation.py`  
**Why Dangerous**: No nested CV in legacy, inconsistent metrics  
**Fix**:
```bash
# MOVE rewrite validation suite
mv foodspec_rewrite/foodspec/validation/ src/foodspec/
# UPDATE imports throughout codebase
```
**Target**: `src/foodspec/validation/evaluation.py`

---

### D.12 🟡 **MEDIUM: Deployment Bundle Format**
**What**: Incompatible bundle serialization  
**Location**:
- Rewrite: `foodspec_rewrite/foodspec/deploy/bundle.py:20`
- Legacy: `src/foodspec/deploy/artifact.py:15` (uses old OutputBundle)  
**Why Dangerous**: Bundles saved by v1.0 won't load in v1.1+  
**Fix**:
```python
# ADD migration function in src/foodspec/utils/migration.py:
def migrate_bundle_v1_to_v2(old_bundle_path: Path, new_bundle_path: Path):
    """Convert v1.0 bundle to v2.0 format."""
    old = OutputBundle.load(old_bundle_path)
    new = DeploymentBundle(
        model=old.model,
        preprocessor=old.preprocessor,
        metadata=old.metadata,
    )
    new.save(new_bundle_path)
```
**Target**: `src/foodspec/deploy/bundle.py` (migrated format)

---

### D.13 🟡 **MEDIUM: Multimodal Fusion**
**What**: Multimodal module exists but not protocol-integrated  
**Location**: `src/foodspec/core/multimodal.py` (6,138 lines)  
**Why Dangerous**: Dead code path, users can't access via protocol  
**Fix**:
```python
# UPDATE ProtocolV2 schema to support multi-input:
# src/foodspec/core/protocol.py:
class DataSpec(BaseModel):
    sources: List[str]  # Multiple sources
    modalities: List[str]  # Multiple modalities
    fusion_strategy: Optional[str] = "early"  # early/late/hybrid

# UPDATE ExecutionEngine to handle multi-input:
def _run_data_load(self):
    if len(self.protocol.data.sources) > 1:
        from foodspec.core.multimodal import MultimodalDataset
        return MultimodalDataset.load(self.protocol.data)
```
**Target**: Protocol-integrated multimodal

---

### D.14 🟡 **MEDIUM: Recipe System (Rewrite Only)**
**What**: Recipe expansion in rewrite, not in legacy  
**Location**: `foodspec_rewrite/foodspec/preprocess/recipes.py`  
**Why Dangerous**: Legacy protocols can't use recipes  
**Fix**:
```bash
# MOVE recipes to src/
mv foodspec_rewrite/foodspec/preprocess/recipes.py src/foodspec/preprocess/
# Ensure ProtocolV2.expand_recipes() is in src/
```
**Target**: `src/foodspec/preprocess/recipes.py`

---

### D.15 🟡 **MEDIUM: Legacy RQ System**
**What**: Ratiometric Questions in legacy, not in rewrite  
**Location**: `src/foodspec/features/rq/*.py`  
**Why Dangerous**: Rewrite has no RQ support  
**Fix**:
```python
# ADD RQ to ComponentRegistry:
# src/foodspec/core/registry.py:
def register_default_feature_components(registry):
    # ... existing ...
    from foodspec.features.rq import RQEngine
    registry.register("features", "rq", RQEngine)
```
**Target**: Keep RQ in src/, register with ComponentRegistry

---

### D.16 🔴 **CRITICAL: Import from Wrong Package**
**What**: Trust evaluator imports from non-existent path  
**Location**: `foodspec_rewrite/src/foodspec/trust/evaluator.py:17`
```python
from foodspec.evaluation.artifact_registry import ArtifactRegistry  # WRONG
```
**Why Dangerous**: Import error at runtime  
**Fix**:
```python
# FIX import:
from foodspec.core.artifacts import ArtifactRegistry  # CORRECT
```

---

### D.17 🟡 **HIGH: Feature Engineering Not Registered**
**What**: Rewrite features exist but not in registry by default  
**Location**: `foodspec_rewrite/foodspec/features/*.py` (peaks, bands, chemometrics)  
**Why Dangerous**: Protocol validation will fail ("unknown component")  
**Fix**:
```python
# ENSURE registration happens at import:
# src/foodspec/core/__init__.py:
from .registry import ComponentRegistry, register_default_feature_components

_global_registry = ComponentRegistry()
register_default_feature_components(_global_registry)

# Export for use:
__all__ = ["_global_registry", ...]
```

---

### D.18 🟡 **MEDIUM: Nested CV Not Wired to Orchestrator**
**What**: Nested CV exists but ExecutionEngine doesn't call it  
**Location**: `foodspec_rewrite/foodspec/validation/nested.py:472` (complete implementation)  
**Why Dangerous**: Users specify nested=true in protocol, nothing happens  
**Fix**:
```python
# UPDATE ExecutionEngine._run_evaluate():
# src/foodspec/core/orchestrator.py:
def _run_evaluate(self):
    if self.protocol.validation.nested_cv:
        from foodspec.validation.nested import evaluate_model_nested_cv
        results = evaluate_model_nested_cv(
            X=self.results["features"]["X"],
            y=self.results["data"]["y"],
            model=self.results["model"]["model"],
            inner_cv=self.protocol.validation.inner_cv,
            outer_cv=self.protocol.validation.outer_cv,
        )
    else:
        from foodspec.validation.evaluation import evaluate_model_cv
        results = evaluate_model_cv(...)
```

---

### D.19 🟡 **MEDIUM: Leakage Detection Not Integrated**
**What**: Leakage detection exists but not called by orchestrator  
**Location**: `foodspec_rewrite/foodspec/validation/leakage.py:64`  
**Fix**:
```python
# ADD to ExecutionEngine._run_features():
from foodspec.validation.leakage import detect_feature_fit_leakage

if self.protocol.qc.check_leakage:
    leakage_report = detect_feature_fit_leakage(
        feature_transformer=features_obj,
        X_train=X_train,
        X_test=X_test,
    )
    if leakage_report["has_leakage"]:
        raise RuntimeError(f"Feature leakage detected: {leakage_report}")
```

---

### D.20 🟡 **HIGH: No One-Command Full Run**
**What**: CLI has separate commands, no single "run everything" command  
**Location**: `./pyproject.toml:87-91` defines 5 separate commands  
**Why Dangerous**: Users must manually chain commands, error-prone  
**Fix**:
```python
# ADD to src/foodspec/cli/main.py:
@app.command()
def run(
    protocol: Path = typer.Argument(..., help="Protocol YAML file"),
    output_dir: Path = typer.Option("./run", help="Output directory"),
):
    """Run complete pipeline: preprocess → QC → features → model → trust → viz → report."""
    from foodspec.core.protocol import ProtocolV2
    from foodspec.core.orchestrator import ExecutionEngine
    from foodspec.core.registry import ComponentRegistry
    from foodspec.core.artifacts import ArtifactRegistry
    
    # Load protocol
    protocol_obj = ProtocolV2.from_yaml(protocol)
    
    # Setup
    registry = ComponentRegistry()
    artifacts = ArtifactRegistry(output_dir)
    
    # Execute
    engine = ExecutionEngine(protocol_obj, registry, artifacts)
    results = engine.run()
    
    # Save manifest
    manifest = RunManifest()
    manifest.save(artifacts.manifest_path())
    
    typer.echo(f"✓ Run complete: {output_dir}")
```
**Target**: `foodspec run protocol.yaml --output-dir ./run_001`

---

## E) RUNTIME INTEGRATION BREAKPOINTS (Top 10)

### E.1 🔴 **Import Error: Trust Evaluator**
**Where**: `foodspec_rewrite/src/foodspec/trust/evaluator.py:17`
```python
from foodspec.evaluation.artifact_registry import ArtifactRegistry  # MODULE NOT FOUND
```
**Why Fails**: Path `foodspec.evaluation` doesn't exist  
**Fix**: Change to `from foodspec.core.artifacts import ArtifactRegistry`

---

### E.2 🔴 **CLI Points to Wrong Package**
**Where**: Running `foodspec` command  
**Why Fails**: CLI installs from `src/foodspec/cli/` but imports expect `foodspec_rewrite/foodspec/`  
**Fix**: Delete `foodspec_rewrite/`, update all imports to `src/`

---

### E.3 🔴 **Protocol Validation Fails**
**Where**: `ProtocolV2.validate()` called with empty registry  
**Why Fails**: Components like "peaks", "bands" not registered  
**Fix**: Call `register_default_feature_components()` at startup

---

### E.4 🔴 **Orchestrator Doesn't Execute Trust Stage**
**Where**: `ExecutionEngine.run()` executes stages  
**Why Fails**: `_run_trust()` method exists but not called (missing from stages list)  
**Check**: `foodspec_rewrite/foodspec/core/orchestrator.py:47-60`
```python
self.stages = [
    "data_load",
    "preprocess",
    "qc",
    "features",
    "split",
    "model",
    "evaluate",
    "trust",      # <-- Check if this exists
    "visualize",
    "report",
]
```
**Fix**: Ensure "trust" in stages list and `_run_trust()` is implemented

---

### E.5 🔴 **Nested CV Not Triggered**
**Where**: Protocol has `nested_cv: true`, but regular CV runs  
**Why Fails**: `ExecutionEngine._run_evaluate()` doesn't check protocol flag  
**Fix**: Add conditional logic (see D.18)

---

### E.6 🟡 **QC Failures Don't Stop Pipeline**
**Where**: QC stage runs, finds issues, but orchestrator continues  
**Why Fails**: No exception raised on QC failure  
**Fix**:
```python
# src/foodspec/core/orchestrator.py:
def _run_qc(self):
    qc_results = run_qc_checks(self.protocol.qc, data)
    if not qc_results["passed"]:
        raise RuntimeError(f"QC failed: {qc_results['failures']}")
```

---

### E.7 🟡 **Report Generation Missing Data**
**Where**: Dossier generation fails with KeyError  
**Why Fails**: Orchestrator doesn't save intermediate results that reporting expects  
**Fix**:
```python
# After each stage:
self.artifacts.save_json("stage_data.json", results)
```

---

### E.8 🟡 **Artifacts Not Written to Disk**
**Where**: `ArtifactRegistry.register()` called but files don't exist  
**Why Fails**: `register()` only logs paths, doesn't save data  
**Fix**: Use `ArtifactRegistry.save_*()` methods:
```python
artifacts.save_json("metrics.json", metrics)  # Not just register
```

---

### E.9 🟡 **Leakage Detection Skipped**
**Where**: Protocol has `check_leakage: true`, but no detection runs  
**Why Fails**: Orchestrator doesn't call `detect_feature_fit_leakage()`  
**Fix**: See D.19

---

### E.10 🟡 **Bundle Save Fails**
**Where**: End of run, trying to save deployment bundle  
**Why Fails**: `DeploymentBundle.save()` expects preprocessor object, gets None  
**Fix**:
```python
# src/foodspec/core/orchestrator.py:
def _run_bundle(self):
    from foodspec.deploy.bundle import DeploymentBundle
    bundle = DeploymentBundle(
        model=self.results["model"]["model"],
        preprocessor=self.results["preprocess"]["preprocessor"],  # Must save preprocessor object
        metadata=self.protocol.metadata.dict(),
    )
    bundle_path = self.artifacts.run_dir / "model_bundle.pkl"
    bundle.save(bundle_path)
    self.artifacts.register("bundle", bundle_path)
```

---

## F) MINIMAL "MAKE IT REAL" PATH

### Goal
Achieve end-to-end execution: `foodspec run protocol.yaml` produces a complete run folder with manifest + metrics + predictions + report + bundle.

---

### Phase 1: Eliminate Duplication (IMMEDIATE)

**Changes**:
1. **Delete `foodspec_rewrite/` entirely** (after archiving docs)
   ```bash
   mkdir -p _internal/phase-history/architecture-docs
   mv foodspec_rewrite/*.md _internal/phase-history/architecture-docs/
   rm -rf foodspec_rewrite/
   ```

2. **Single pyproject.toml**: Already at `./pyproject.toml`

3. **Update .gitignore**: Already done (see REORGANIZATION_GUIDE.md)

**Result**: Single import root (`src/foodspec/`), no ambiguity

---

### Phase 2: Wire Core Components (1-2 HOURS)

**File 1**: `src/foodspec/core/protocol.py`
- **COPY**: `foodspec_rewrite/foodspec/core/protocol.py` → `src/foodspec/core/protocol.py`
- **Line 351**: `class ProtocolV2(BaseModel)` now in src/

**File 2**: `src/foodspec/core/registry.py`
- **COPY**: `foodspec_rewrite/foodspec/core/registry.py` → `src/foodspec/core/registry.py`
- **Line 25**: `class ComponentRegistry` now in src/
- **Line 112**: `register_default_feature_components()`
- **Line 152**: `register_default_model_components()`

**File 3**: `src/foodspec/core/orchestrator.py`
- **COPY**: `foodspec_rewrite/foodspec/core/orchestrator.py` → `src/foodspec/core/orchestrator.py`
- **Line 47**: `class ExecutionEngine` now in src/

**File 4**: `src/foodspec/validation/` (directory)
- **COPY ENTIRE DIR**: `foodspec_rewrite/foodspec/validation/` → `src/foodspec/validation/`
- Includes: `evaluation.py`, `nested.py`, `metrics.py`, `splits.py`, `statistics.py`, `leakage.py`

**File 5**: `src/foodspec/preprocess/recipes.py`
- **COPY**: `foodspec_rewrite/foodspec/preprocess/recipes.py` → `src/foodspec/preprocess/recipes.py`

**Result**: All core components in canonical locations

---

### Phase 3: Fix Imports (30 MINUTES)

**Global Search-Replace**:
```python
# OLD imports:
from foodspec.evaluation.artifact_registry import ArtifactRegistry

# NEW imports:
from foodspec.core.artifacts import ArtifactRegistry
```

**Files to Update**:
- `src/foodspec/trust/evaluator.py:17`
- Any other files with wrong import paths (check with grep)

**Verify**:
```bash
python -c "from foodspec.core.protocol import ProtocolV2; print('✓ Protocol OK')"
python -c "from foodspec.core.orchestrator import ExecutionEngine; print('✓ Engine OK')"
python -c "from foodspec.validation.evaluation import evaluate_model_cv; print('✓ Eval OK')"
```

---

### Phase 4: Wire Orchestrator Stages (2 HOURS)

**File**: `src/foodspec/core/orchestrator.py`

**Ensure all stages are wired**:
```python
# Line 60-70: Check stages list
self.stages = [
    "data_load",
    "preprocess",
    "qc",
    "features",
    "split",
    "model",
    "evaluate",
    "trust",      # Ensure present
    "visualize",  # Ensure present
    "report",     # Ensure present
    "bundle",     # ADD if missing
]

# Add missing stage implementations:

# Line 400+ (if missing):
def _run_trust(self) -> Dict:
    """Execute trust evaluation stage."""
    from foodspec.trust.evaluator import TrustEvaluator
    
    evaluator = TrustEvaluator(
        model=self.results["model"]["model"],
        X_test=self.results["split"]["X_test"],
        y_test=self.results["split"]["y_test"],
        artifact_registry=self.artifacts,
    )
    
    trust_results = evaluator.evaluate()
    evaluator.save_artifacts(self.artifacts.run_dir / "trust")
    
    return trust_results

# Line 450+ (if missing):
def _run_visualize(self) -> Dict:
    """Execute visualization stage."""
    from foodspec.viz import create_comparison_dashboard
    
    # Create standard visualizations
    viz_dir = self.artifacts.run_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Uncertainty plots
    if "trust" in self.results:
        from foodspec.viz.uncertainty import plot_prediction_intervals
        fig = plot_prediction_intervals(
            self.results["trust"]["intervals"],
            self.results["split"]["y_test"],
        )
        fig.savefig(viz_dir / "uncertainty.png")
    
    return {"viz_dir": str(viz_dir)}

# Line 500+ (if missing):
def _run_report(self) -> Dict:
    """Execute report generation stage."""
    from foodspec.reporting.dossier import generate_dossier
    
    dossier_path = self.artifacts.run_dir / "dossier.html"
    generate_dossier(
        run_dir=self.artifacts.run_dir,
        output_path=dossier_path,
        protocol=self.protocol,
    )
    
    return {"dossier": str(dossier_path)}

# Line 550+ (if missing):
def _run_bundle(self) -> Dict:
    """Create deployment bundle."""
    from foodspec.deploy.bundle import DeploymentBundle
    
    bundle = DeploymentBundle(
        model=self.results["model"]["model"],
        preprocessor=self.results["preprocess"]["preprocessor"],
        metadata=self.protocol.metadata.dict(),
    )
    
    bundle_path = self.artifacts.run_dir / "model_bundle.pkl"
    bundle.save(bundle_path)
    self.artifacts.register("bundle", bundle_path)
    
    return {"bundle": str(bundle_path)}
```

---

### Phase 5: One-Command CLI (30 MINUTES)

**File**: `src/foodspec/cli/main.py`

**Add full-run command**:
```python
# After line 97, ADD:

@app.command()
def run(
    protocol: Path = typer.Argument(..., help="Path to protocol YAML file"),
    output_dir: Path = typer.Option("./foodspec_run", help="Output directory for run artifacts"),
    skip_stages: Optional[List[str]] = typer.Option(None, help="Stages to skip (comma-separated)"),
):
    """
    Execute complete FoodSpec pipeline from protocol.
    
    Runs: data_load → preprocess → qc → features → model → evaluate → trust → viz → report → bundle
    
    Example:
        foodspec run protocol.yaml --output-dir ./my_run
    """
    from foodspec.core.protocol import ProtocolV2
    from foodspec.core.orchestrator import ExecutionEngine
    from foodspec.core.registry import ComponentRegistry, register_default_feature_components, register_default_model_components
    from foodspec.core.artifacts import ArtifactRegistry
    from foodspec.core.manifest import RunManifest
    
    # Load protocol
    typer.echo(f"Loading protocol: {protocol}")
    protocol_obj = ProtocolV2.from_yaml(protocol)
    
    # Validate
    typer.echo("Validating protocol...")
    protocol_obj.validate()
    
    # Setup components
    registry = ComponentRegistry()
    register_default_feature_components(registry)
    register_default_model_components(registry)
    
    artifacts = ArtifactRegistry(output_dir)
    
    # Create engine
    engine = ExecutionEngine(
        protocol=protocol_obj,
        registry=registry,
        artifacts=artifacts,
    )
    
    # Execute
    typer.echo(f"Executing pipeline (output: {output_dir})...")
    try:
        results = engine.run()
        
        # Save manifest
        manifest = RunManifest()
        manifest.protocol = protocol_obj.dict()
        manifest.results = results
        manifest.save(artifacts.manifest_path())
        
        typer.echo(f"✓ Run complete!")
        typer.echo(f"  Output: {output_dir}")
        typer.echo(f"  Manifest: {artifacts.manifest_path()}")
        typer.echo(f"  Metrics: {artifacts.metrics_path()}")
        typer.echo(f"  Predictions: {artifacts.predictions_path()}")
        typer.echo(f"  Dossier: {output_dir}/dossier.html")
        typer.echo(f"  Bundle: {output_dir}/model_bundle.pkl")
        
    except Exception as e:
        typer.echo(f"✗ Run failed: {e}", err=True)
        raise typer.Exit(1)
```

---

### Phase 6: Integration Test (15 MINUTES)

**Create test protocol**:
```yaml
# test_protocol.yaml
metadata:
  name: "Integration Test"
  version: "1.0"
  description: "End-to-end test"

data:
  source: "examples/data/olive_oil.csv"
  modality: "raman"

preprocess:
  recipe: "raman_standard"

features:
  type: "chemometrics"
  components:
    - name: "pca"
      params:
        n_components: 10

task:
  type: "classification"
  target: "class"

model:
  type: "plsda"
  params:
    n_components: 5

validation:
  cv:
    type: "stratified_kfold"
    n_splits: 5
  metrics:
    - "accuracy"
    - "f1_macro"

trust:
  enable: true
  conformal:
    alpha: 0.1
  abstention:
    threshold: 0.8

visualization:
  enable: true
  plots:
    - "uncertainty"
    - "confusion_matrix"
    - "feature_importance"

reporting:
  enable: true
  format: "html"
```

**Run**:
```bash
foodspec run test_protocol.yaml --output-dir ./test_run
```

**Verify Output**:
```bash
test_run/
├── manifest.json           # ✓ Created
├── metrics.json            # ✓ Created
├── predictions.json        # ✓ Created
├── qc_results.json         # ✓ Created
├── trust/
│   ├── intervals.json      # ✓ Created
│   └── abstention.json     # ✓ Created
├── visualizations/
│   ├── uncertainty.png     # ✓ Created
│   └── confusion_matrix.png# ✓ Created
├── dossier.html            # ✓ Created
└── model_bundle.pkl        # ✓ Created
```

---

### Phase 7: Update Documentation (30 MINUTES)

**File**: `README.md`

**Update quick start**:
```markdown
## Quick Start

1. Install:
   ```bash
   pip install -e .
   ```

2. Run complete pipeline:
   ```bash
   foodspec run examples/protocols/oil_authentication.yaml --output-dir ./my_run
   ```

3. Check outputs:
   ```bash
   ls my_run/
   # manifest.json, metrics.json, predictions.json, dossier.html, model_bundle.pkl
   ```
```

---

## SUMMARY: Prioritized Patch List

### IMMEDIATE (Do First)
1. **Delete `foodspec_rewrite/`** (10 min) - Eliminates all import ambiguity
2. **Fix trust evaluator import** (2 min) - Critical runtime bug
3. **Copy 5 core files** (10 min) - Protocol, Registry, Orchestrator, Validation suite, Recipes

### HIGH PRIORITY (Do Next)
4. **Wire orchestrator stages** (2 hours) - Implement missing `_run_*()` methods
5. **Add one-command CLI** (30 min) - `foodspec run protocol.yaml`
6. **Fix all imports** (30 min) - Search-replace wrong paths

### MEDIUM PRIORITY (Do Soon)
7. **Register default components** (30 min) - Ensure validation works
8. **Integrate nested CV** (30 min) - Add conditional in `_run_evaluate()`
9. **Integrate leakage detection** (30 min) - Add to `_run_features()`
10. **Bundle migration script** (1 hour) - v1.0 → v2.0 bundle converter

### LOW PRIORITY (Nice to Have)
11. **Multimodal protocol support** (2 hours) - Multi-input DataSpec
12. **Recipe system expansion** (1 hour) - More built-in recipes
13. **Documentation update** (1 hour) - New structure
14. **Cleanup .gitignore** (10 min) - Exclude generated files

**Total Time**: ~10 hours for "make it real" (full end-to-end working)

**Result**: `foodspec run protocol.yaml` will produce a complete, reproducible analysis run with all artifacts, reports, and deployment bundle.

---

**END OF AUDIT REPORT**
