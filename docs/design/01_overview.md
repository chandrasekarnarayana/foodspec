# foodspec: Gold-Standard Spectroscopy Toolkit for Food Science

## 1. Vision and scope
- Deliver a reproducible, production-ready Python toolkit for Raman and FTIR spectroscopy in food science, spanning pre-processing, feature extraction, chemometrics/ML, and domain modules.
- Serve both research and industrial pipelines: from exploratory analysis of single spectra to automated batch processing of hyperspectral maps in QA/QC and surveillance.
- Integrate with spectral libraries and curated example datasets to anchor methods in referenceable ground truth.
- Prioritize transparency and auditability: configuration-driven pipelines, provenance capture, and shareable analysis artifacts (HDF5/NetCDF).

## 2. Key user personas and use cases
- **Analytical chemist (R&D):** Explore spectra, apply baseline/ATR/atmospheric corrections, compare to reference libraries, extract peak ratios for authenticity checks, publishable figures.
- **Food quality engineer (production/QC):** Run validated pipelines on batches/hyperspectral maps, flag outliers or adulteration, generate reports and trend dashboards, integrate with LIMS/MES.
- **Data scientist (ML/chemometrics):** Build PCA/PLS/PLS-DA/SVM/RF/XGBoost models, optimize preprocessing chains, perform one-class novelty detection, deploy models as reproducible configs.
- **Regulatory/compliance analyst:** Trace model versions and datasets, reproduce decisions, export locked-down reports and provenance trails.
- **Academia/education:** Use example datasets and notebooks to teach spectroscopy fundamentals, band assignments, and chemometric workflows.

## 3. Scientific requirements (spectroscopy + food science)
- **Pre-processing:** Baseline correction (polynomial/AsLS/airPLS), smoothing (Savitzky-Golay), normalization (vector/area/peak), derivatives (1st/2nd), ATR correction, atmospheric (CO₂/H₂O) removal, cosmic ray handling.
- **Feature extraction:** Peak finding (adaptive prominence/width), band area/integration, peak ratios, spectral descriptors (centroid, FWHM, skewness, kurtosis), region-of-interest utilities.
- **Chemometrics and ML:** PCA, clustering (k-means/hierarchical/DBSCAN), PLS/PLS-DA, SVM, random forest, XGBoost, linear/nonlinear regression, one-class methods, mixture analysis (NNLS, MCR-ALS).
- **Data modalities:** Single spectra and hyperspectral maps (Raman/FTIR imaging); support wavelength/wavenumber axes, metadata (instrument, acquisition params), and mask handling.
- **Food-specific modules:** Edible oils adulteration and grading, oil uptake in fried products, heating/degradation markers, dairy/meat authentication, microbial detection, moisture/fat/protein estimation.
- **Libraries/datasets:** Interfaces to spectral libraries; packaged example datasets for each application module with metadata and expected outputs for tests/tutorials.

## 4. Software requirements (architecture, quality, performance)
- **Architecture:** Modular packages for `io`, `preprocess`, `features`, `chemometrics`, `maps`, `applications`, `pipelines`, and `viz`; composable, stateless functions with optional pipeline objects/configs.
- **Data model:** Core spectral object for 1D spectra with axes, units, metadata; hyperspectral cube type with spatial axes and masks; immutable data classes with type hints and validation.
- **APIs:** Functional API for building chains; declarative configs (YAML/JSON) to define pipelines; I/O to HDF5/NetCDF for reproducibility; adapters for numpy/pandas/xarray.
- **Quality:** Full type hints, docstrings, doctests; unit/integration tests with fixtures for spectra and maps; CI running lint (ruff/flake8), type-check (mypy/pyright), tests, coverage, and docs build.
- **Performance:** Vectorized operations; optional numba/Cython for hot paths; chunked processing for hyperspectral maps; graceful degradation with progress reporting and caching.
- **Docs and UX:** User guides, API reference, cookbook notebooks; reproducible examples; CLI entry points for common tasks; plotting utilities for spectra, residuals, score/loadings, and map overlays.

## 5. High-level architecture (modules, data model, pipelines)
- **Core types:** `Spectrum1D` (intensity, axis, units, metadata), `HyperMap` (cube with spatial dims, mask), `Provenance` (pipeline steps, versions), `Dataset` (collections with labels/splits).
- **I/O (`io`):** Read/write common vendor-neutral formats and exports to HDF5/NetCDF; dataset loaders for packaged examples; library adapters.
- **Pre-processing (`preprocess`):** Baseline, smoothing, normalization, derivatives, ATR/atmospheric corrections, cosmic ray removal; step objects with fit/transform for reproducibility.
- **Feature extraction (`features`):** Peak detection, band areas, ratios, spectral descriptors; ROI utilities and feature tables.
- **Chemometrics/ML (`chemometrics`):** PCA, clustering, PLS/PLS-DA, SVM, RF, XGBoost, regression, one-class methods, NNLS/MCR-ALS for mixtures; compatible with scikit-learn interfaces.
- **Maps (`maps`):** Hyperspectral processing (per-pixel or block), masking, spectral flattening, and map-level visualizations; memory-aware chunking.
- **Pipelines/configs (`pipelines`):** Declarative pipeline builder consuming configs; stores provenance; exports/imports pipeline definitions; deterministic seeding and version stamping.
- **Applications (`applications`):** Domain-specific workflows and reference configs for oils, oil-in-chips, heating degradation, dairy/meat authentication, microbial detection; bundled examples and benchmarks.
- **Visualization (`viz`):** Plotting helpers for spectra, baselines, residuals, PCA/PLS scores/loadings, clustering maps, classification outputs, and hyperspectral overlays.

## 6. Roadmap phases
- **Current (v0.2.x):** Core data models; broad preprocessing suite (baseline, smoothing, normalization, derivatives, ATR/atmospheric, cosmic ray handling); peak/band/ratio features; PCA/PLS/PLS-DA and mixture tools; stats (parametric/nonparametric/robustness); metrics and viz; domain workflows (oils/heating/mixtures/QC); troubleshooting, reproducibility, and developer standards; optional vendor loaders and DL prototypes.
- **v0.3 (near-term):** Harden vendor IO (SPC/OPUS/JCAMP) with examples; expand interpretability visuals (RF/PLS loadings); add public dataset fixtures/benchmarks; refine DL calibration examples and cautions; raise coverage on IO/stats edge branches; optional lint/type CI.
- **v0.4–v0.9:** Performance and usability polish (chunked hyperspectral pipelines, optional acceleration), richer domain templates (dairy/spices/beverages), more real-data tutorials, expanded robustness/uncertainty utilities, streamlined config-driven pipelines and reporting.
- **v1.0:** Stability and API freeze; complete vendor/data adapters and library management; validated benchmarks on public datasets; deployment guides; documented migration/deprecation paths; reproducibility guarantees (seeds, configs, provenance) and long-term support stance.
