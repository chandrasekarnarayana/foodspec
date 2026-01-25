# FoodSpec Canonical Module Map

**Target State**: Single coherent architecture with `src/foodspec/` as canonical root.

---

## Directory Structure (Final)

```
src/foodspec/
├── __init__.py                    # Package root + version
├── core/                          # Core architecture
│   ├── protocol.py               # ProtocolV2 (Pydantic schema)
│   ├── registry.py               # ComponentRegistry + defaults
│   ├── orchestrator.py           # ExecutionEngine + Orchestrator
│   ├── artifacts.py              # ArtifactRegistry (standard paths)
│   ├── manifest.py               # RunManifest (provenance)
│   └── output_bundle.py          # Backwards compat (deprecated)
├── io/                            # Data loading & serialization
│   ├── __init__.py
│   ├── loaders.py                # CSV/TXT/JCAMP/HDF5 loaders
│   ├── ingest.py                 # Format detection
│   ├── export.py                 # Save to standard formats
│   └── serialization.py          # JSON/pickle helpers
├── preprocess/                    # Spectroscopic preprocessing
│   ├── __init__.py
│   ├── baseline.py               # 6 baseline removal methods
│   ├── smoothing.py              # Savitzky-Golay, moving average
│   ├── normalization.py          # Vector, SNV, MSC
│   ├── derivatives.py            # 1st/2nd derivatives
│   ├── ftir.py                   # ATR/atmospheric correction
│   ├── recipes.py                # Protocol-driven preprocessing sequences
│   └── utils.py                  # Helpers
├── qc/                            # Quality Control
│   ├── __init__.py
│   ├── drift.py                  # Temporal drift detection
│   ├── leakage.py                # Feature fit leakage detection
│   ├── governance.py             # Data governance checks
│   ├── engine.py                 # QC orchestrator
│   └── reports.py                # QC summary generation
├── features/                      # Feature Engineering
│   ├── __init__.py
│   ├── peaks.py                  # Peak detection
│   ├── bands.py                  # Band integration
│   ├── chemometrics.py           # PCA, PLS loadings, VIP
│   ├── rq/                        # Ratiometric Questions (legacy)
│   │   ├── __init__.py
│   │   ├── engine.py             # RQ execution
│   │   └── library.py            # Pre-built RQs
│   ├── selection.py              # Feature selection (SelectKBest, RFE)
│   ├── marker_panel.py           # Link to domain markers
│   └── utils.py                  # Helpers
├── models/                        # ML Model Training
│   ├── __init__.py
│   ├── classical.py              # PLS-DA, PLS-R, SVM, RF, Logistic
│   ├── boosting.py               # XGBoost, LightGBM
│   ├── neural.py                 # Neural nets (if added)
│   ├── calibration.py            # Model-level calibration
│   └── utils.py                  # Helpers
├── validation/                    # Model Evaluation & Validation
│   ├── __init__.py
│   ├── splits.py                 # CV splitters (stratified, time-series, etc.)
│   ├── nested.py                 # Nested cross-validation
│   ├── evaluation.py             # evaluate_model, evaluate_model_cv, etc.
│   ├── metrics.py                # 20+ metrics (acc, F1, ROC-AUC, etc.)
│   ├── statistics.py             # Statistical tests
│   ├── leakage.py                # Feature fit leakage detection
│   └── utils.py                  # Helpers
├── trust/                         # Trust & Uncertainty Quantification
│   ├── __init__.py
│   ├── conformal.py              # Conformal prediction (ICP, cross-conformal)
│   ├── calibration.py            # Platt scaling, isotonic regression
│   ├── abstain.py                # Confidence-based abstention
│   ├── coverage.py               # Coverage guarantee validation
│   ├── reliability.py            # Reliability curves & diagrams
│   ├── evaluator.py              # TrustEvaluator (orchestrator)
│   └── utils.py                  # Helpers
├── viz/                           # Visualization
│   ├── __init__.py
│   ├── compare.py                # Multi-run comparison dashboard
│   ├── uncertainty.py            # Uncertainty/confidence plots
│   ├── embeddings.py             # t-SNE, UMAP visualizations
│   ├── processing_stages.py      # Preprocessing pipeline viz
│   ├── coefficients.py           # Model coefficients heatmaps
│   ├── stability.py              # CV fold stability plots
│   ├── paper.py                  # Publication-ready presets (JOSS/Nature)
│   └── utils.py                  # Helpers
├── reporting/                     # Report Generation
│   ├── __init__.py
│   ├── dossier.py                # Comprehensive HTML dossier
│   ├── pdf.py                    # PDF export (WeasyPrint)
│   ├── export.py                 # Archive bundles (zip)
│   ├── base.py                   # Base Reporter class
│   ├── cards.py                  # Individual report cards
│   ├── engine.py                 # Report orchestration
│   └── templates/                # HTML/CSS templates
│       ├── base.html
│       ├── metrics.html
│       └── style.css
├── deploy/                        # Deployment & Inference
│   ├── __init__.py
│   ├── bundle.py                 # DeploymentBundle (v2.0 format)
│   ├── predict.py                # Inference pipeline
│   ├── migration.py              # v1.0 → v2.0 bundle migration
│   └── utils.py                  # Helpers
├── cli/                           # Command-line Interface
│   ├── __init__.py
│   ├── main.py                   # Main entry point, foodspec run
│   ├── utils.py                  # CLI helpers
│   └── commands/                 # (Optional) Subcommands
│       ├── run.py
│       ├── validate.py
│       └── export.py
├── core/                          # (legacy, deprecated)
│   ├── api.py                    # High-level unified API (legacy, phase out)
│   ├── dataset.py                # SpectralDataset (legacy, phase out)
│   ├── multimodal.py             # Multimodal fusion (keep but not primary)
│   └── hyperspectral.py          # Hyperspectral support (keep)
├── utils/                         # Shared Utilities
│   ├── __init__.py
│   ├── logging.py                # Logging configuration
│   ├── migration.py              # v1.0 → v2.0 migrations
│   ├── config.py                 # Configuration management
│   └── helpers.py                # General helpers
└── __version__.py                # Version constant

examples/
├── quickstarts/                   # Single-file, <5 min examples
│   ├── oil_authentication.py     # Quick classification demo
│   ├── aging.py                  # Aging prediction
│   ├── heating_quality.py        # Heat quality classification
│   └── mixture_analysis.py       # Mixture component prediction
├── protocols/                     # Full protocol YAML files
│   ├── oil_authentication_full.yaml
│   ├── aging_full.yaml
│   ├── heating_quality_full.yaml
│   └── test_minimal.yaml         # For CI testing
├── notebooks/                     # Interactive Jupyter notebooks
│   ├── 01_intro.ipynb
│   ├── 02_data_exploration.ipynb
│   └── 03_workflow.ipynb
├── advanced/                      # Complex, multi-file examples
│   ├── multimodal_fusion_demo.py
│   ├── vip_demo.py
│   ├── governance_demo.py
│   ├── validation_chemometrics.py
│   └── README.md
├── configs/                       # Configuration templates
│   ├── oil_raman.yaml
│   ├── oil_ftir.yaml
│   └── oil_hyperspectral.yaml
├── data/                          # Small test datasets
│   ├── olive_oil_sample.csv
│   ├── heating_sample.csv
│   └── README.md (links to large data)
└── fixtures/                      # Data fixtures for tests
    ├── small_raman.csv
    └── small_ftir.csv

tests/
├── test_architecture.py           # NEW: Architecture enforcement tests
├── test_architecture_ci.py        # NEW: CI-level tests
├── unit/
│   ├── test_protocol.py
│   ├── test_registry.py
│   ├── test_orchestrator.py
│   ├── test_evaluation.py
│   └── ...
├── integration/
│   ├── test_minimal_e2e.py       # NEW: Minimal end-to-end test
│   ├── test_full_pipeline.py
│   └── ...
└── fixtures/
    ├── minimal_protocol.yaml      # NEW: Minimal test protocol
    └── sample_data.csv

.github/workflows/
├── tests.yml                      # Existing unit tests
├── architecture-enforce.yml       # NEW: Architecture enforcement
└── e2e.yml                        # NEW: End-to-end verification

scripts/
├── refactor_executor.py           # NEW: Phased refactoring executor
├── validate_architecture.py       # NEW: Architecture validation
├── check_docs_links.py            # Existing
├── test_examples_imports.py       # Existing
└── ...

```

---

## Module Responsibilities

### `core/` - Architecture Foundation
- **protocol.py**: ProtocolV2 (Pydantic validation + expansion)
- **registry.py**: ComponentRegistry + default registration (features, models, splitters)
- **orchestrator.py**: ExecutionEngine (stage-by-stage execution)
- **artifacts.py**: ArtifactRegistry (standard run directory layout)
- **manifest.py**: RunManifest (provenance, checksums, version tracking)

### `io/` - Data I/O
- Load: CSV, TXT, JCAMP, HDF5, MAT, SPC
- Detect format automatically
- Save results: JSON, CSV, HDF5
- Serialization helpers (pickle, JSON with custom encoders)

### `preprocess/` - Preprocessing Pipeline
- Baseline removal: 6 methods (ALS, rubberband, polynomial, airPLS, modified poly, rolling ball)
- Smoothing: Savitzky-Golay, moving average
- Normalization: Vector, SNV, MSC
- Derivatives: 1st, 2nd
- FTIR-specific: ATR correction, atmospheric compensation
- **recipes.py**: Protocol-driven preprocessing chains (e.g., "raman_standard")

### `qc/` - Quality Control
- Drift detection: Temporal trends, batch effects
- Leakage detection: Feature-fit leakage in CV
- Governance: Traceability, reproducibility checks
- Orchestrator: Runs all QC checks in sequence
- Reporting: Summary metrics

### `features/` - Feature Engineering
- **peaks.py**: Peak detection (prominence, width filtering)
- **bands.py**: Band integration (normalized region areas)
- **chemometrics.py**: PCA loadings, PLS weights, VIP
- **rq/**: Ratiometric Questions (legacy domain knowledge)
- **selection.py**: SelectKBest, RFE, L1-based selection
- **marker_panel.py**: Link features to domain markers (e.g., "iron content")

### `models/` - Model Training
- **classical.py**: PLS-DA, PLS-R, SVM, Random Forest, Logistic Regression
- **boosting.py**: XGBoost, LightGBM
- **calibration.py**: Model-level output calibration
- Support vector: One-vs-rest, probability estimates

### `validation/` - Model Evaluation
- **splits.py**: Stratified K-fold, time-series CV, group CV, LOO
- **nested.py**: Nested CV (outer for test, inner for hyperparameter tuning)
- **evaluation.py**: evaluate_model, evaluate_model_cv, evaluate_model_nested_cv
- **metrics.py**: Accuracy, F1, ROC-AUC, PR-AUC, confusion matrix, etc. (20+ metrics)
- **statistics.py**: T-tests, McNemar, sign test, bootstrap confidence intervals
- **leakage.py**: Detect feature-fit leakage in cross-validation

### `trust/` - Trust & Uncertainty
- **conformal.py**: Split conformal, ICP, cross-conformal prediction
- **calibration.py**: Platt scaling, isotonic regression
- **abstain.py**: Abstention based on confidence threshold
- **coverage.py**: Validate empirical coverage vs. desired level
- **reliability.py**: Reliability curves, Brier score
- **evaluator.py**: TrustEvaluator (orchestrates all trust metrics)

### `viz/` - Visualization
- **compare.py**: Multi-run comparison dashboard
- **uncertainty.py**: Prediction intervals, confidence bands
- **embeddings.py**: t-SNE, UMAP 2D/3D projections
- **processing_stages.py**: Data transformation heatmaps
- **coefficients.py**: Model coefficients heatmaps (with significance)
- **stability.py**: CV fold stability, consensus heatmaps
- **paper.py**: Publication presets (JOSS, Nature, Science)

### `reporting/` - Report Generation
- **dossier.py**: Comprehensive HTML dossier (metrics + viz + metadata)
- **pdf.py**: PDF export with styling
- **export.py**: Zip archive with all artifacts
- **cards.py**: Individual report cards (summary + detailed)
- **engine.py**: Report orchestration (what to include, format)

### `deploy/` - Deployment
- **bundle.py**: DeploymentBundle (v2.0 format: model + preprocessor + metadata)
- **predict.py**: Inference pipeline (apply preprocessing + model)
- **migration.py**: v1.0 bundle → v2.0 bundle conversion
- Support batch and streaming inference

### `cli/` - Command-Line Interface
- **main.py**: Entry point, `foodspec run` command
- One unified command: `foodspec run protocol.yaml --output-dir ./run`
- Minimal CLI, protocol-driven

---

## Key Principles

1. **Single Source of Truth**:
   - Only ONE `src/foodspec/` package root
   - All code flows from this root
   - No reachable duplicates

2. **Protocol-Driven**:
   - All behavior configurable via ProtocolV2 YAML
   - No hardcoded defaults (use registry)
   - Expansible via registry registration

3. **Orchestrated Execution**:
   - ExecutionEngine runs stages in sequence
   - Each stage is independent (loose coupling)
   - Results passed via `self.results` dict

4. **Reproducibility**:
   - RunManifest captures: protocol, data hash, versions, environment
   - Artifacts registered in standard paths
   - Bundle includes preprocessing + model + metadata

5. **Extensibility**:
   - ComponentRegistry maps strings to classes
   - Default registration at startup
   - Custom components via plugin registration

6. **Trust by Default**:
   - Conformal prediction available in every run
   - Calibration automatic
   - Abstention optional but recommended

---

## Import Patterns

### Public API (intended usage)
```python
# Protocol definition
from foodspec.core.protocol import ProtocolV2
protocol = ProtocolV2.from_yaml("protocol.yaml")

# Execution
from foodspec.core.orchestrator import ExecutionEngine
from foodspec.core.registry import ComponentRegistry
from foodspec.core.artifacts import ArtifactRegistry

registry = ComponentRegistry()
artifacts = ArtifactRegistry("./run")
engine = ExecutionEngine(protocol, registry, artifacts)
results = engine.run()
```

### Internal (within foodspec)
```python
# Cross-module imports within src/foodspec/
from foodspec.core.artifacts import ArtifactRegistry
from foodspec.validation.evaluation import evaluate_model_cv
from foodspec.trust.evaluator import TrustEvaluator
```

### Deprecated (phase out)
```python
# These should be removed or redirected:
from foodspec.core.api import FoodSpecAPI
from foodspec.protocol.config import ProtocolConfig  # Use ProtocolV2
from foodspec.protocol.runner import ProtocolRunner  # Use ExecutionEngine
```

---

## NO OTHER DIRECTORY LAYOUTS ARE ALLOWED

This map is canonical. Any deviation signals:
- Import ambiguity (violates single source of truth)
- Duplicate implementations (maintenance burden)
- Unclear module boundaries (architectural debt)

**Enforcement**: CI tests will fail if:
- `foodspec_rewrite/` exists
- Multiple `pyproject.toml` files exist
- Imports reference non-canonical paths
- Missing expected modules in final layout

---

**END OF CANONICAL MODULE MAP**
