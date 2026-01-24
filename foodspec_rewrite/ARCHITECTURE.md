# FoodSpec 2.0 Architecture Guide

## Clean Architecture Rewrite

This is the skeleton for FoodSpec 2.0: a complete architectural rewrite using clean code principles and protocol-driven design.

### Directory Structure Overview

```
foodspec_rewrite/
│
├── foodspec/                  # Main package
│   ├── __init__.py           # Package initialization, clean exports
│   │
│   ├── core/                 # 🔷 Core protocols & infrastructure
│   │   ├── __init__.py
│   │   ├── registry.py       # Component registry (extensibility)
│   │   ├── orchestrator.py   # Workflow orchestration & composition
│   │   ├── manifest.py       # Reproducibility metadata
│   │   ├── artifacts.py      # Output collection & serialization
│   │   └── cache.py          # Performance caching layer
│   │
│   ├── io/                   # 📂 Data I/O
│   │   ├── __init__.py
│   │   ├── loaders.py        # Load from files/folders
│   │   ├── formats.py        # Format detection & conversion
│   │   └── library.py        # Library management
│   │
│   ├── preprocess/           # 🔧 Data preprocessing
│   │   ├── __init__.py
│   │   ├── baseline.py       # Baseline correction (ALS, polynomial, rubberband)
│   │   ├── normalize.py      # Normalization methods
│   │   └── harmonize.py      # Dataset harmonization & alignment
│   │
│   ├── qc/                   # ✓ Quality control
│   │   ├── __init__.py
│   │   ├── checks.py         # QC checks (balance, outliers, missing)
│   │   ├── validators.py     # Data validators
│   │   └── reports.py        # QC reporting
│   │
│   ├── features/             # 🎯 Feature extraction
│   │   ├── __init__.py
│   │   ├── spectral.py       # Peak detection, ratios, areas
│   │   ├── statistical.py    # Mean, variance, entropy, etc.
│   │   └── domain.py         # Domain-specific features
│   │
│   ├── models/               # 🤖 Machine learning
│   │   ├── __init__.py
│   │   ├── base.py           # Base model classes
│   │   ├── sklearn_models.py # Scikit-learn wrappers
│   │   ├── xgboost_models.py # XGBoost wrappers
│   │   └── keras_models.py   # Deep learning models
│   │
│   ├── validation/           # 📊 Model validation
│   │   ├── __init__.py
│   │   ├── splitters.py      # Train/test splitting
│   │   ├── cross_val.py      # Cross-validation
│   │   └── metrics.py        # Evaluation metrics
│   │
│   ├── trust/                # 🔐 Uncertainty & trustworthiness
│   │   ├── __init__.py
│   │   ├── uncertainty.py    # Confidence intervals, std errors
│   │   ├── calibration.py    # Probability calibration
│   │   └── robustness.py     # Robustness checks
│   │
│   ├── viz/                  # 📈 Visualization
│   │   ├── __init__.py
│   │   ├── plots.py          # Matplotlib plots
│   │   ├── interactive.py    # Plotly/Bokeh interactive
│   │   └── style.py          # Common styling
│   │
│   ├── reporting/            # 📄 Report generation
│   │   ├── __init__.py
│   │   ├── templates.py      # Report templates
│   │   ├── export.py         # PDF, HTML, PNG export
│   │   └── formatter.py      # Text/table formatting
│   │
│   ├── deploy/               # 🚀 Model deployment
│   │   ├── __init__.py
│   │   ├── server.py         # FastAPI/Flask server
│   │   ├── batch.py          # Batch prediction
│   │   └── serving.py        # Model serving utilities
│   │
│   └── cli/                  # 💻 Command-line interface
│       ├── __init__.py
│       ├── main.py           # CLI entry point (Typer)
│       └── commands/
│           ├── __init__.py
│           ├── preprocess.py # Preprocessing commands
│           ├── train.py      # Training commands
│           ├── analyze.py    # Analysis commands
│           └── serve.py      # Deployment commands
│
├── tests/                    # 🧪 Tests
│   ├── __init__.py
│   ├── test_core.py         # Core protocol tests
│   ├── test_io.py           # I/O tests
│   ├── test_preprocess.py   # Preprocessing tests
│   ├── test_models.py       # Model tests
│   ├── test_integration.py  # Integration tests
│   └── fixtures.py          # Shared test fixtures
│
├── docs/                     # 📚 Documentation
│   ├── index.md             # Main doc index
│   ├── architecture.md      # Architecture guide
│   ├── api/                 # API documentation
│   ├── tutorials/           # User tutorials
│   └── examples/            # Example notebooks
│
├── examples/                 # 💡 Example code
│   ├── quickstart.py        # Quick start example
│   ├── preprocessing.py     # Preprocessing examples
│   ├── training.py          # Training examples
│   └── deployment.py        # Deployment examples
│
├── pyproject.toml           # Project configuration
└── README.md                # Project README
```

---

## Core Design Patterns

### 1. **Protocol-Based Design** (Not Inheritance)

```python
# In core/__init__.py
from typing import Protocol

class Spectrum(Protocol):
    @property
    def wavenumbers(self) -> list[float]: ...
    
    @property
    def intensities(self) -> list[float]: ...

# Any object with these properties satisfies the protocol
# No explicit inheritance needed—structural typing
```

**Benefits:**
- Duck typing with type safety
- Loose coupling between components
- Easy to test with mocks
- No deep inheritance hierarchies

### 2. **Registry Pattern** (Extensibility)

```python
# In core/registry.py
from typing import Type, Dict, Any

class Registry:
    """Extensible component registry."""
    
    def __init__(self):
        self._components: Dict[str, Type] = {}
    
    def register(self, name: str, cls: Type):
        """Register a new component."""
        self._components[name] = cls
    
    def get(self, name: str, **kwargs) -> Any:
        """Instantiate a registered component."""
        cls = self._components[name]
        return cls(**kwargs)

# Usage
registry = Registry()
registry.register("baseline_als", BaselineALS)
registry.register("baseline_poly", BaselinePolynomial)

baseline = registry.get("baseline_als", method="symmetric")
```

**Benefits:**
- Add new components without modifying existing code
- Runtime component selection
- Plugin architecture support
- Configuration-driven workflows

### 3. **Orchestrator Pattern** (Workflow Composition)

```python
# In core/orchestrator.py
class Orchestrator:
    """Compose and execute workflows."""
    
    def __init__(self):
        self.steps = []
    
    def add(self, name: str, step):
        """Add a workflow step."""
        self.steps.append((name, step))
        return self
    
    def run(self, data):
        """Execute workflow."""
        result = data
        for name, step in self.steps:
            result = step(result)
            print(f"✓ {name}")
        return result

# Usage
workflow = Orchestrator()
workflow.add("load", LoadData(path))
workflow.add("preprocess", Preprocess(method="als"))
workflow.add("extract", FeatureExtraction(features=["ratio_1030_1050"]))
workflow.add("train", TrainModel(algorithm="RandomForest"))
result = workflow.run()
```

**Benefits:**
- Declarative workflow definition
- Reusable pipeline components
- Easy to serialize/deserialize
- Reproducibility tracking

### 4. **Artifact-Based Outputs** (Reproducibility)

```python
# In core/artifacts.py
class ArtifactBundle:
    """Collect outputs for reproducibility."""
    
    def __init__(self):
        self.artifacts = {}
    
    def add(self, name: str, obj, metadata: dict = None):
        """Add artifact."""
        self.artifacts[name] = {
            "object": obj,
            "metadata": metadata or {}
        }
    
    def save(self, path: str):
        """Serialize all artifacts."""
        # Save as JSON, pickle, or HDF5
        pass
    
    def load(self, path: str):
        """Load artifacts."""
        pass

# Usage
artifacts = ArtifactBundle()
artifacts.add("model", trained_model, {"framework": "sklearn"})
artifacts.add("metrics", {"accuracy": 0.95, "f1": 0.93})
artifacts.add("manifest", {"date": "2025-01-24", "version": "2.0.0"})
artifacts.save("./outputs/exp_001/")
```

**Benefits:**
- Complete provenance tracking
- Easy experiment comparison
- Reproducible results
- Audit trail for compliance

---

## Module Responsibilities

### `core/` — Protocols & Infrastructure
- Protocol definitions (Spectrum, SpectralDataset, Preprocessor, etc.)
- Registry for component discovery
- Orchestrator for workflow composition
- Manifest for metadata tracking
- ArtifactBundle for output collection
- Cache layer for performance

### `io/` — Data Loading
- Load from various formats (CSV, HDF5, NetCDF, etc.)
- Format auto-detection
- Library management (curated spectral libraries)
- Metadata parsing

### `preprocess/` — Data Transformation
- Baseline correction (ALS, polynomial, rubberband)
- Normalization (L2, mean centering, etc.)
- Harmonization (aligning datasets)
- Resampling/interpolation

### `qc/` — Quality Control
- Data validation (missing values, outliers)
- Class balance checks
- Replicate consistency
- Dataset readiness scoring

### `features/` — Feature Extraction
- Peak detection and characterization
- Peak ratios (e.g., 1030/1050 cm⁻¹)
- Statistical features (entropy, kurtosis)
- Domain-specific features (oil authentication, etc.)

### `models/` — Machine Learning
- Wrapper classes for sklearn, XGBoost, Keras
- Training, prediction, evaluation
- Model serialization/deserialization
- Hyperparameter optimization

### `validation/` — Model Validation
- Train/test splitting (with stratification)
- K-fold cross-validation
- Evaluation metrics (accuracy, F1, ROC, etc.)
- Leakage detection

### `trust/` — Uncertainty & Trustworthiness
- Confidence intervals
- Calibration analysis
- Robustness checks
- Adversarial testing

### `viz/` — Visualization
- Spectral plots with preprocessing overlays
- PCA/clustering visualization
- Model performance plots
- Interactive dashboards

### `reporting/` — Report Generation
- HTML/PDF reports
- Experiment summaries
- Methodology documentation
- Results tables and figures

### `deploy/` — Model Serving
- FastAPI server for predictions
- Batch prediction utility
- Docker containerization
- Kubernetes deployment

### `cli/` — Command-Line Interface
- Typer-based CLI
- Common workflow commands (preprocess, train, predict)
- Plugin system for custom commands

---

## Development Workflow

### 1. Define Protocol (in `core/`)
```python
# foodspec/core/__init__.py
class MyComponent(Protocol):
    def process(self, data) -> OutputData: ...
```

### 2. Implement Concrete Classes
```python
# foodspec/module/implementation.py
class ConcreteComponent:
    def process(self, data) -> OutputData:
        # Implementation
        pass
```

### 3. Register Component (optional)
```python
# In module initialization
registry.register("my_component", ConcreteComponent)
```

### 4. Add Tests
```python
# tests/test_module.py
def test_component():
    comp = ConcreteComponent()
    output = comp.process(test_data)
    assert output.is_valid()
```

### 5. Document in README
```markdown
## MyModule

### Usage
```python
comp = ConcreteComponent()
result = comp.process(data)
```
```

---

## Next Steps

1. **Implement core protocols** in `foodspec/core/__init__.py`
2. **Build I/O layer** in `foodspec/io/`
3. **Implement preprocessing** in `foodspec/preprocess/`
4. **Add QC checks** in `foodspec/qc/`
5. **Extract features** in `foodspec/features/`
6. **Add models** in `foodspec/models/`
7. **Implement validation** in `foodspec/validation/`
8. **Add uncertainty** in `foodspec/trust/`
9. **Create visualizations** in `foodspec/viz/`
10. **Generate reports** in `foodspec/reporting/`
11. **Deploy models** in `foodspec/deploy/`
12. **Build CLI** in `foodspec/cli/`
13. **Write comprehensive tests** in `tests/`
14. **Document everything** in `docs/`

---

## Key Files to Complete

### Immediate (Core)
- [ ] `foodspec/core/registry.py` — Component registry
- [ ] `foodspec/core/orchestrator.py` — Workflow engine
- [ ] `foodspec/core/manifest.py` — Metadata tracking
- [ ] `foodspec/core/artifacts.py` — Output serialization
- [ ] `foodspec/core/cache.py` — Caching layer

### Short-term (Essential)
- [ ] `foodspec/io/loaders.py` — Data loading
- [ ] `foodspec/preprocess/baseline.py` — Baseline correction
- [ ] `foodspec/qc/checks.py` — Quality checks
- [ ] `foodspec/features/spectral.py` — Peak extraction
- [ ] `foodspec/models/base.py` — Model base classes

### Medium-term (Enhancement)
- [ ] `foodspec/validation/metrics.py` — Evaluation metrics
- [ ] `foodspec/trust/uncertainty.py` — Confidence intervals
- [ ] `foodspec/viz/plots.py` — Visualization
- [ ] `foodspec/reporting/templates.py` — Report generation
- [ ] `foodspec/deploy/server.py` — API server

---

## Testing Strategy

```python
# tests/test_core.py
import pytest
from foodspec.core import Registry, Orchestrator

def test_registry():
    """Registry can register and retrieve components."""
    reg = Registry()
    reg.register("test", TestComponent)
    comp = reg.get("test", param="value")
    assert isinstance(comp, TestComponent)

def test_orchestrator():
    """Orchestrator chains steps correctly."""
    orch = Orchestrator()
    orch.add("step1", Step1())
    orch.add("step2", Step2())
    result = orch.run(initial_data)
    assert result.is_valid()
```

---

## References

- **Clean Architecture**: Robert C. Martin
- **Design Patterns**: Gang of Four
- **Python Protocols**: PEP 544
- **Domain-Driven Design**: Eric Evans
- **Testing in Python**: pytest docs
- **Type Hints**: Python typing module

---

## Status

✅ **Skeleton created** with:
- All directories set up
- Core protocols defined
- Example CLI implemented
- pyproject.toml configured
- README with architecture overview

🚀 **Ready to implement** module by module following the architecture guide above.
