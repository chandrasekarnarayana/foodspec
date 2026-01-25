# Implementation Checklist: FoodSpec 2.0

## ✅ Phase 0: Skeleton (COMPLETE)

- [x] Directory structure created
- [x] Core protocols defined (in `foodspec/core/__init__.py`)
- [x] CLI entry point created (`foodspec/cli/main.py`)
- [x] Project configuration (`pyproject.toml`)
- [x] Architecture documentation (`ARCHITECTURE.md`)
- [x] Quick-start example (`examples/quickstart.py`)
- [x] README with overview

---

## 📋 Phase 1: Core Infrastructure

### core/registry.py
- [ ] Registry class with register/get methods
- [ ] Support for lazy instantiation
- [ ] Built-in components (baseline, preprocessing, models)
- [ ] Error handling for missing components
- [ ] Tests in `tests/test_core.py`

### core/orchestrator.py
- [ ] Orchestrator class with add/run methods
- [ ] Step execution with error handling
- [ ] Manifest capture during execution
- [ ] Artifact collection
- [ ] Tests for chaining and composition

### core/manifest.py
- [ ] Manifest class for metadata
- [ ] Timestamp, version, parameters tracking
- [ ] Serialization (to_dict/from_dict)
- [ ] JSON export

### core/artifacts.py
- [ ] ArtifactBundle class
- [ ] Add/save/load methods
- [ ] Support for multiple formats (pickle, JSON, HDF5)
- [ ] Metadata preservation
- [ ] Tests for serialization

### core/cache.py
- [ ] Cache class for performance
- [ ] LRU cache implementation
- [ ] Invalidation strategies
- [ ] Optional caching decorator

---

## 📂 Phase 2: Data I/O

### io/loaders.py
- [ ] Load from CSV, JSON
- [ ] Load from HDF5, NetCDF
- [ ] Load from folders
- [ ] Auto-detection of format
- [ ] Metadata parsing
- [ ] Tests with sample data

### io/formats.py
- [ ] Format detection (magic bytes, extension)
- [ ] Format-specific parsers
- [ ] Header extraction
- [ ] Standardization to (wavenumbers, spectra, metadata)

### io/library.py
- [ ] Library management
- [ ] Load standard libraries (FTIR, Raman)
- [ ] Library search (by sample type, modality)
- [ ] Library comparison

---

## 🔧 Phase 3: Preprocessing

### preprocess/baseline.py
- [ ] BaselineALS implementation
- [ ] Polynomial baseline
- [ ] Rubberband baseline
- [ ] Tests with reference data
- [ ] Performance benchmarks

### preprocess/normalize.py
- [ ] L2 normalization
- [ ] Mean centering
- [ ] Min-max scaling
- [ ] Area normalization
- [ ] Tests and comparisons

### preprocess/harmonize.py
- [ ] Dataset alignment
- [ ] Wavenumber interpolation
- [ ] Multi-dataset harmonization
- [ ] Tests with misaligned datasets

---

## ✓ Phase 4: Quality Control

### qc/checks.py
- [ ] Missing value detection
- [ ] Outlier detection (IQR, LOF)
- [ ] Class balance checks
- [ ] Replicate consistency
- [ ] Tests for each check

### qc/validators.py
- [ ] Data validators
- [ ] Schema validation
- [ ] Pydantic models for config
- [ ] Custom validation rules

### qc/reports.py
- [ ] QC report generation
- [ ] Summary statistics
- [ ] Visualizations
- [ ] Recommendations

---

## 🎯 Phase 5: Feature Extraction

### features/spectral.py
- [ ] Peak detection (find_peaks)
- [ ] Peak characterization (height, width, area)
- [ ] Peak ratios (e.g., 1030/1050)
- [ ] Wavenumber-specific features
- [ ] Tests with known peaks

### features/statistical.py
- [ ] Mean, median, std, variance
- [ ] Entropy, kurtosis, skewness
- [ ] Quantiles, percentiles
- [ ] Range statistics

### features/domain.py
- [ ] Oil authentication features
- [ ] Heating quality indicators
- [ ] Microbial indicators
- [ ] Custom domain features

---

## 🤖 Phase 6: Machine Learning

### models/base.py
- [ ] BaseModel class
- [ ] Fit/predict interface
- [ ] Serialization support
- [ ] Parameter management

### models/sklearn_models.py
- [ ] Wrapper for RandomForest, SVM, LogisticRegression
- [ ] Grid search support
- [ ] Cross-validation
- [ ] Model comparison

### models/xgboost_models.py
- [ ] XGBoost classifier/regressor
- [ ] Hyperparameter optimization
- [ ] Feature importance

### models/keras_models.py
- [ ] Neural network models
- [ ] Transfer learning
- [ ] Fine-tuning

---

## 📊 Phase 7: Validation

### validation/splitters.py
- [ ] Train/test split (with stratification)
- [ ] K-fold cross-validation
- [ ] Time-series split
- [ ] Stratified splits

### validation/cross_val.py
- [ ] Cross-validation runners
- [ ] Score aggregation
- [ ] Fold reporting

### validation/metrics.py
- [ ] Classification metrics (accuracy, F1, precision, recall)
- [ ] Regression metrics (MAE, RMSE, R²)
- [ ] ROC/AUC curves
- [ ] Confusion matrix

---

## 🔐 Phase 8: Uncertainty & Trust

### trust/uncertainty.py
- [ ] Confidence intervals
- [ ] Standard error calculation
- [ ] Bootstrap confidence intervals
- [ ] Prediction intervals

### trust/calibration.py
- [ ] Probability calibration
- [ ] Reliability plots
- [ ] Calibration metrics

### trust/robustness.py
- [ ] Perturbation testing
- [ ] Adversarial examples
- [ ] Model stability checks

---

## 📈 Phase 9: Visualization

### viz/plots.py
- [ ] Spectral plots with preprocessing overlays
- [ ] Comparison plots
- [ ] Baseline visualization
- [ ] Feature importance plots

### viz/interactive.py
- [ ] Plotly dashboards
- [ ] Interactive spectrum exploration
- [ ] Model performance dashboard

### viz/style.py
- [ ] Matplotlib style setup
- [ ] Color palettes
- [ ] Common formatting

---

## 📄 Phase 10: Reporting

### reporting/templates.py
- [ ] Report template classes
- [ ] Methodology sections
- [ ] Results sections
- [ ] Recommendations

### reporting/export.py
- [ ] Export to PDF
- [ ] Export to HTML
- [ ] Export to PNG (figures)
- [ ] Multi-format support

### reporting/formatter.py
- [ ] Table formatting
- [ ] Statistics formatting
- [ ] Rounding and precision

---

## 🚀 Phase 11: Deployment

### deploy/server.py
- [ ] FastAPI application
- [ ] POST /predict endpoint
- [ ] GET /health endpoint
- [ ] Model loading
- [ ] Request validation

### deploy/batch.py
- [ ] Batch prediction utility
- [ ] Parallel processing
- [ ] Progress tracking
- [ ] Error handling

### deploy/serving.py
- [ ] Docker containerization
- [ ] Environment setup
- [ ] Model versioning
- [ ] API documentation

---

## 💻 Phase 12: CLI

### cli/main.py
- [x] Basic Typer setup
- [ ] Expand with subcommands
- [ ] Plugin system
- [ ] Help documentation

### cli/commands/
- [ ] preprocess.py — Preprocessing commands
- [ ] train.py — Training commands
- [ ] analyze.py — Analysis commands
- [ ] serve.py — Deployment commands
- [ ] export.py — Export commands

---

## 🧪 Phase 13: Testing

### tests/
- [ ] test_core.py — Registry, Orchestrator, Manifest
- [ ] test_io.py — Loaders, format detection
- [ ] test_preprocess.py — Baseline, normalization
- [ ] test_qc.py — QC checks and validators
- [ ] test_features.py — Feature extraction
- [ ] test_models.py — Model training and prediction
- [ ] test_validation.py — Validation and metrics
- [ ] test_trust.py — Uncertainty quantification
- [ ] test_viz.py — Visualization
- [ ] test_integration.py — End-to-end workflows

**Coverage Target:** ≥85%

---

## 📚 Phase 14: Documentation

### docs/
- [ ] Architecture guide (ARCHITECTURE.md done ✓)
- [ ] API reference (auto-generated from docstrings)
- [ ] User guide with examples
- [ ] Tutorial: Classification workflow
- [ ] Tutorial: Regression workflow
- [ ] Tutorial: Custom preprocessing
- [ ] FAQ
- [ ] Troubleshooting

### Code Documentation
- [ ] NumPy-style docstrings
- [ ] Type hints throughout
- [ ] Examples in docstrings
- [ ] Module-level documentation

---

## ✨ Phase 15: Polish & Release

- [ ] Code review and refactoring
- [ ] Performance profiling
- [ ] Documentation review
- [ ] Security audit
- [ ] Version bumping (2.0.0-alpha → 2.0.0)
- [ ] Changelog
- [ ] Release notes
- [ ] PyPI upload
- [ ] GitHub release

---

## Progress Summary

| Phase | Component | Status |
|-------|-----------|--------|
| 0 | Skeleton | ✅ DONE |
| 1 | Core Infrastructure | ⏳ TODO |
| 2 | Data I/O | ⏳ TODO |
| 3 | Preprocessing | ⏳ TODO |
| 4 | Quality Control | ⏳ TODO |
| 5 | Feature Extraction | ⏳ TODO |
| 6 | Machine Learning | ⏳ TODO |
| 7 | Validation | ⏳ TODO |
| 8 | Uncertainty | ⏳ TODO |
| 9 | Visualization | ⏳ TODO |
| 10 | Reporting | ⏳ TODO |
| 11 | Deployment | ⏳ TODO |
| 12 | CLI | ⏳ TODO |
| 13 | Testing | ⏳ TODO |
| 14 | Documentation | ⏳ TODO |
| 15 | Polish & Release | ⏳ TODO |

---

## Quick Start for Contributors

```bash
# 1. Navigate to skeleton
cd foodspec_rewrite/

# 2. Install for development
pip install -e ".[dev]"

# 3. Pick a phase and component
# e.g., Implement core/registry.py

# 4. Write tests first (TDD)
# tests/test_core.py

# 5. Run tests
pytest tests/test_core.py -v --cov=foodspec/core

# 6. Document with docstrings
# Use NumPy style: https://numpydoc.readthedocs.io/

# 7. Format code
black foodspec/
ruff check foodspec/
mypy foodspec/

# 8. Create PR
git push -u origin phase-2/core-infrastructure
# Create pull request with phase/component details
```

---

## Notes

- Each phase can be implemented in parallel once dependencies are clear
- Tests should be written before implementation (TDD)
- Documentation is mandatory for all public APIs
- Code reviews required before merge
- Performance benchmarks for critical paths
- Security audit before public release

---

## References

- **Architecture**: See ARCHITECTURE.md
- **Protocols**: See foodspec/core/__init__.py
- **Examples**: See examples/quickstart.py
- **Config**: See pyproject.toml
