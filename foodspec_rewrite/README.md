# FoodSpec 2.0: Clean Architecture Rewrite

**Protocol-driven spectroscopy framework for food science**

A complete architectural rewrite of FoodSpec with focus on:
- **Protocol-based design** (duck typing, structural typing)
- **Registry pattern** (extensibility without modification)
- **Orchestrator pattern** (workflow composition)
- **Artifact-based** (reproducibility, serialization)
- **Clean code** (SOLID principles, type hints, testing)

## Directory Structure

```
foodspec/
  core/       Protocol definitions, registry, orchestrator, manifest
  io/         Data loading, format detection, library management
  preprocess/ Baseline correction, harmonization, normalization
  qc/         Quality control, outlier detection
  features/   Feature extraction (spectral, statistical, domain-specific)
  models/     ML models, training, prediction
  validation/ Cross-validation, stratification
  trust/      Uncertainty quantification
  viz/        Plotting, interactive visualization
  reporting/  Report templates, export
  deploy/     Model serving, API
  cli/        Command-line interface
tests/        Unit and integration tests
docs/         Documentation, API reference
examples/     Jupyter notebooks, use cases
```

## Key Design Principles

### 1. Protocol-Based (Not Inheritance)
```python
class Spectrum(Protocol):
    @property
    def wavenumbers(self) -> list[float]: ...
    @property
    def intensities(self) -> list[float]: ...

# Any object with these properties satisfies Spectrum
```

### 2. Registry Pattern
```python
registry = Registry()
registry.register("baseline_als", BaselineALS)
baseline = registry.get("baseline_als")
```

### 3. Orchestrator Pattern
```python
workflow = Orchestrator()
workflow.add("load", LoadData(path))
workflow.add("preprocess", Preprocess(method="als"))
workflow.add("extract", FeatureExtraction())
result = workflow.run()
```

### 4. Artifact-Based Outputs
```python
artifacts = ArtifactBundle()
artifacts.add("model", trained_model)
artifacts.add("metrics", evaluation_metrics)
artifacts.add("manifest", metadata)
artifacts.save("./outputs/")
artifacts.load("./outputs/")
```

## Development

### Install for Development
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest tests/ -v --cov=foodspec
```

### Format Code
```bash
black foodspec/
ruff check foodspec/
mypy foodspec/
```

## Roadmap

- [x] Architecture design
- [x] Core protocols
- [ ] I/O layer
- [ ] Preprocessing pipeline
- [ ] QC checks
- [ ] Feature extraction
- [ ] Model management
- [ ] Validation framework
- [ ] Uncertainty quantification
- [ ] Visualization
- [ ] Reporting
- [ ] Deployment
- [ ] CLI
- [ ] Documentation

## References

- Clean Architecture (Robert C. Martin)
- Design Patterns (Gang of Four)
- Python Protocol (PEP 544)
- Domain-Driven Design (Eric Evans)
