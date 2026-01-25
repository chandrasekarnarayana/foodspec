# Changelog


## [1.2.0] - Unreleased

### Added
- Mindmap-aligned namespaces (`data_objects`, `engine`, `modeling`) with compatibility shims
- Design philosophy documentation and mindmap mapping table
- IO validators and loader wrappers for Raman/FTIR/vendor inputs
- CLI groups for `io`, `qc`, `preprocess`, `features`, `train`
- Reproducibility snapshot helper and model/dataset card scaffolds
- Pre-commit configuration and formatting/typing configuration

### Changed
- README rewritten to clarify goals, non-goals, and mindmap-aligned features
- CLI `report` command updated to support `--run-dir` reporting

### Deprecated
- None (backward compatibility maintained via wrappers)

## [1.1.0] - 2026-01-25

### Added
- ✨ **New Protocol-Driven Architecture** - Complete rewrite with modern design
- 🛡️  **Trust Subsystem** - Uncertainty quantification and abstention logic
- 📊 **Reporting System** - PDF export, dossiers, and paper presets
- 📈 **Visualization Suite** - Multi-run comparison, uncertainty plots
- 🔧 **Deprecation Warnings** - All legacy code marked for removal in v2.0.0

### Deprecated
- ⚠️  Root-level modules (spectral_dataset.py, output_bundle.py, etc.)
- ⚠️  Old CLI scripts (cli_*.py)
- ⚠️  demo/ package
- ⚠️  report/ package

See `docs/migration/v1-to-v2.md` for migration guide.

### Migration Path
v1.1.0 → v1.2.0 → v1.3.0 → v1.4.0 → v2.0.0 (deprecated code removed)

**Action Required:** Update your code to use new imports before v2.0.0 (planned: June 2026)


All notable changes to FoodSpec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-25

### 🎉 First Production Release

FoodSpec v1.0.0 is now production-ready with comprehensive functionality for food spectroscopy analysis.

#### Package Statistics
- **28,080 lines** of production code
- **95 public APIs** across 23 modules
- **685 passing tests** with 78.57% coverage
- **150+ pages** of documentation
- **16 production-ready examples**

### Added

#### Core Features
- Complete spectral data structures (FoodSpectrumSet, HyperSpectralCube, MultiModalDataset)
- High-level FoodSpec API with scikit-learn style interface
- OutputBundle for reproducible output management
- RunRecord for experiment tracking

#### Preprocessing
- 6 baseline correction methods (ALS, rubberband, polynomial, airpls, modified polynomial, rolling ball)
- 5 normalization methods (vector, area, max, reference, SNV)
- Smoothing (Savitzky-Golay, moving average, Gaussian, median)
- Derivatives (1st/2nd order)
- Scatter correction (MSC, EMSC)
- Spike removal and cosmic ray detection
- Calibration transfer (PDS, slope/bias)

#### Feature Extraction
- Ratiometric Questions (RQ) engine for ratio computation
- Peak statistics and detection
- Spectral fingerprinting and similarity
- PCA/PLS with comprehensive visualization
- VIP scores for variable importance

#### Machine Learning
- 10+ classification algorithms (PLS-DA, SVM, Random Forest, etc.)
- Nested cross-validation for unbiased evaluation
- Calibration diagnostics (Brier, ECE, reliability diagrams)
- Multimodal fusion (late fusion, decision fusion)
- Hyperparameter tuning with grid/random search

#### Statistical Analysis
- Hypothesis tests (t-tests, ANOVA, MANOVA, non-parametric)
- Multiple testing correction (Bonferroni, Benjamini-Hochberg)
- Effect sizes (Cohen's d, eta², omega²)
- Power analysis and sample size calculation
- Correlations (Pearson, Spearman, Kendall)
- Method comparison (Bland-Altman)

#### Quality Control
- Novelty detection (One-Class SVM, Isolation Forest, LOF)
- Drift monitoring with control charts
- Leakage detection
- Dataset validation
- Prediction guards and confidence thresholding

#### Domain Applications
- Edible oil authentication
- Meat quality assessment
- Dairy analysis
- Microbial contamination detection
- Heating/thermal degradation analysis
- Mixture quantification with NNLS

#### Protocol System
- YAML-based workflow definition
- 7 protocol step types (preprocess, RQ, output, harmonize, HSI segment/ROI, QC)
- Configuration validation
- Reproducible execution

#### Input/Output
- Support for 10+ vendor formats (Thermo, Bruker, Agilent, PerkinElmer, etc.)
- HDF5 with schema versioning (v1.1)
- CSV, Excel, JSON support
- Library management utilities

#### Visualization
- Comprehensive spectral plotting
- PCA scores and loadings plots
- Classification visualization (confusion matrices, ROC curves)
- Regression plots (predicted vs actual, residuals)
- Hyperspectral false-color composites

#### CLI & Deployment
- Command-line interface with 6 main commands
- Model registry and versioning
- Artifact system for model deployment
- Plugin system for extensibility

#### Documentation
- 150+ pages across 12 hierarchical levels
- API references with mkdocstrings
- 16 runnable examples with full documentation
- Theory-practice integration
- Comprehensive developer guides

### Known Limitations

These items are scaffolded but not fully implemented (planned for v1.1):
- `qc/health.py` - Spectrum health scoring (SNR, drift, spikes)
- `deploy/predict.py` - Deployed predictor (interface defined, prediction stubbed)
- `workflows/library_search.py` - Spectral library matching (placeholder scoring)

### Deprecations

The following modules are deprecated and will be removed in v2.0:
- `artifact.py` → Use `deploy.save_artifact`, `deploy.load_artifact`
- `calibration_transfer.py` → Use `preprocess.calibration_transfer`
- `heating_trajectory.py` → Use `workflows.heating_trajectory`
- `matrix_correction.py` → Use `preprocess.matrix_correction`
- `protocol_engine.py` → Use `protocol.ProtocolRunner`
- `rq.py` → Use `features.rq`
- `spectral_dataset.py` → Use `core.spectral_dataset`
- `spectral_io.py` → Use `io` module

See MIGRATION_GUIDE.md for migration instructions.

### Performance
- Test suite completes in 133 seconds
- Documentation builds in 15 seconds
- Suitable for datasets up to 10,000 spectra (larger datasets may benefit from chunking)

### Testing
- 685 tests passing (99.4% success rate)
- 78.57% code coverage (exceeds 75% target)
- Integration tests for end-to-end workflows
- 15+ CLI test files

### Contributors
- Chandrasekar Subramani Narayan (@chandrasekarnarayana)

---

## [0.2.1] - 2025-11-30

### Fixed
- Various bug fixes and stability improvements
- Documentation updates

## [0.2.0] - 2025-11-30

### Added
- Initial public release
- Basic spectroscopy functionality
- Core preprocessing pipelines
- Machine learning integration

---

## Release Notes

### Upgrading to v1.0.0

v1.0.0 is a major milestone with significant enhancements. For users on v0.2.x:

1. **Update your imports**: Some modules have been reorganized. Use deprecation warnings as guides.
2. **Review breaking changes**: Check MIGRATION_GUIDE.md for details
3. **Update dependencies**: `pip install --upgrade foodspec`
4. **Run tests**: Verify your workflows still work as expected

### Future Roadmap

- **v1.1** (Q1 2025): Complete scaffold implementations, 80% test coverage
- **v1.2** (Q2 2025): OPLS algorithm, extended format support, cloud storage
- **v1.3** (Q3 2025): GPU acceleration, performance optimization
- **v2.0** (Q4 2025): Deep learning, remove deprecations, breaking changes

See docs/06-developer-guide/GAPS_AND_FUTURE_WORK.md for detailed roadmap.

---

[1.0.0]: https://github.com/chandrasekarnarayana/foodspec/releases/tag/v1.0.0
[0.2.1]: https://github.com/chandrasekarnarayana/foodspec/releases/tag/v0.2.1
[0.2.0]: https://github.com/chandrasekarnarayana/foodspec/releases/tag/V0.2.0
