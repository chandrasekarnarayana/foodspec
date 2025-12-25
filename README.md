# FoodSpec

<p align="left">
  <img src="docs/assets/foodspec_logo.png" alt="FoodSpec logo" width="200">
</p>

[![Tests](https://github.com/chandrasekarnarayana/foodspec/actions/workflows/ci.yml/badge.svg)](https://github.com/chandrasekarnarayana/foodspec/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-79%25-brightgreen)](https://github.com/chandrasekarnarayana/foodspec)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> "Food decides the nature of your mind… Mind is born of the food you take."  
> — Sri Sathya Sai Baba, *Effect of Food on the Mind*, Summer Showers 1993 – Indian Culture and Spirituality (29 May 1993)

Production-ready Python toolkit for Raman/FTIR/NIR spectroscopy in food science. FoodSpec provides reproducible workflows for preprocessing, feature extraction, statistics, and machine learning, with built-in support for protocol-driven analysis, model management, and automated reporting.

**Version:** 1.0.0 | **Status:** Production Ready | **Tests:** 685 passing (79% coverage)

## What problems does FoodSpec solve?

- **Fragmented workflows:** Vendor-specific formats, ad hoc preprocessing, irreproducible scripts.
- **Lack of standards:** No consistent data model across labs or instruments.
- **Manual documentation:** Time-consuming figure and report generation.
- **Reproducibility challenges:** Difficulty sharing, versioning, and archiving analyses.

## What does FoodSpec provide?

### Data & Import
- Unified data model for Raman, FTIR, and NIR spectroscopy
- CSV/TXT/JCAMP loaders; optional vendor support (SPC, OPUS)
- HDF5 spectral libraries for reference materials and calibration

### Preprocessing
- Baseline correction (6 methods: ALS, rubberband, polynomial, airPLS, modified polynomial, rolling ball)
- Smoothing (Savitzky-Golay, moving average), normalization (vector, SNV, MSC)
- Derivatives, cropping, cosmic-ray removal
- ATR/atmospheric correction for FTIR

### Feature Extraction & Interpretation
- Peak, band, and ratio detection with library-based chemical interpretation
- Ratiometric Questions (RQ) engine for reproducible ratio computation
- PCA/PLS scores, fingerprint similarity, VIP scores
- Feature importance summaries with visualization

### Statistics & Quality Control
- Parametric and nonparametric hypothesis tests (t-test, ANOVA, MANOVA, Kruskal-Wallis)
- Bootstrap and permutation-based robustness
- Classification and regression metrics with confidence intervals
- Batch QC, novelty detection, drift monitoring, replicate consistency

### Machine Learning
- Classification: Logistic, SVM, Random Forest, Gradient Boosting, PLS-DA
- Regression: Linear, Partial Least Squares, Random Forest
- XGBoost, LightGBM integration
- Optional Conv1D and MLP deep learning
- Model registry, versioning, and lifecycle tracking

### Domain Workflows
- Oil authentication and quality assessment
- Heating/oxidation trajectory analysis
- Mixture composition estimation (NNLS, MCR-ALS)
- Hyperspectral imaging and spatial mapping
- Calibration transfer between instruments (Direct Standardization, PDS)

### Reproducibility & Reporting
- Protocol-driven execution (YAML configuration)
- Automated narrative reports with metrics, tables, and figures
- Run metadata capture and artifact versioning
- Prediction confidence guards and quality gates
- Full provenance tracking

## Supported modalities

| Modality | Input | Preprocessing | Workflows |
|----------|-------|---|---|
| **Raman** | Vendor/CSV/HDF5 | Baseline, smoothing, cosmic-ray removal | Authentication, heating, mixtures, QC |
| **FTIR** | Vendor/CSV/HDF5 | Baseline, normalization, cropping | Authentication, heating, QC |
| **NIR** | CSV/HDF5 | Smoothing, derivatives, SNV | Calibration, regression, quality |
| **Hyperspectral** | HDF5 | Per-pixel preprocessing | Mapping, segmentation, classification |

## Installation

**Requires Python 3.10 or 3.11**

```bash
pip install foodspec

# Optional extras
pip install 'foodspec[ml]'      # XGBoost, LightGBM
pip install 'foodspec[deep]'    # Conv1D, MLP deep learning
pip install 'foodspec[dev]'     # Documentation, tests, linting
```

See the [inAPI (5 minutes)

```python
from foodspec import load_library
from foodspec.apps.oils import run_oil_authentication_workflow

# Load example dataset
fs = load_library("oils_demo.h5")

# Run complete authentication workflow
result = run_oil_authentication_workflow(fs, label_column="oil_type")

# Access results
print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}")
print(f"Confusion Matrix:\n{result.confusion_matrix}")
```

### CLI (5 minutes)

```bash
# Convert CSV to HDF5 library
foodspec csv-to-library raw_spectra.csv library.h5 \
  --wavenumber-col wavenumber --sample-id-col sample_id

# Run oil authentication workflow
foodspec oil-auth library.h5 \
  --label oil_type --output results/
📚 **[Full Documentation](https://chandrasekarnarayana.github.io/foodspec/)** | 150+ pages of guides, tutorials, and API reference

### Quick Links

- **Getting Started**
  - [Installation Guide](https://chandrasekarnarayana.github.io/foodspec/01-getting-started/installation/)
  - [Python Quickstart](https://chandrasekarnarayana.github.io/foodspec/01-getting-started/quickstart_python/)
  - [CLI Quickstart](https://chandrasekarnarayana.github.io/foodspec/01-getting-started/quickstart_cli/)
  - [15-Minute Tutorial](https://chandrasekarnarayana.github.io/foodspec/01-getting-started/quickstart_15min/)

- **User Guides**
  - [Data Libraries](https://chandrasekarnarayana.github.io/foodspec/04-user-guide/libraries/)
  - [CSV Import](https://chandrasekarnarayana.github.io/foodspec/04-user-guide/csv_to_library/)
  - [CLI Reference](https://chandrasekarnarayana.github.io/foodspec/04-user-guide/cli/)
  -Package Statistics

- **21,500+ lines** of production code
- **685 tests** with 79% coverage
- **150+ documentation pages** with examples
- **16 example scripts** + 3 Jupyter notebooks
- **10+ vendor formats** supported
- **6 baseline methods**, 4 normalization methods
- **10+ ML algorithms** with nested cross-validation
- **Protocol-driven workflows** for reproducibility
:

```bibtex
@software{foodspec2025,
  title = {FoodSpec: Pyt! Please:

- Read the [Contributing Guide](https://chandrasekarnarayana.github.io/foodspec/06-developer-guide/contributing/)
- Follow the [Documentation Guidelines](https://chandrasekarnarayana.github.io/foodspec/06-developer-guide/documentation_guidelines/)
- Write clear code with docstrings and type hints
- Add tests for new features (pytest)
- Ensure all checks pass: `pytest`, `ruff check`, `mkdocs build`

Open issues and pull requests with clear, concise descriptions. We appreciate bug reports, feature requests, and documentation improvement

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata
## Development & Testing

```bash
# Run tests with coverage
pytest --cov=src/foodspec tests/ --cov-report=html

# Lint code
ruff check src/ tests/

# Build documentation
mkdocs build

# Run example scripts
python examples/phase1_quickstart.py
```

All tests passing ✅ | Coverage: 79% ✅ | Documentation builds ✅
- **Advanced Topics**
  - [Model Registry](https://chandrasekarnarayana.github.io/foodspec/05-advanced-topics/model_registry/)
  - [HSI & Harmonization](https://chandrasekarnarayana.github.io/foodspec/05-advanced-topics/hsi_and_harmonization/)
  - [MOATS Implementation](https://chandrasekarnarayana.github.io/foodspec/05-advanced-topics/MOATS_IMPLEMENTATION/)

- **Developer Resources**
  - [Contributing](https://chandrasekarnarayana.github.io/foodspec/06-developer-guide/contributing/)
  - [Documentation Guidelines](https://chandrasekarnarayana.github.io/foodspec/06-developer-guide/documentation_guidelines/)
  - [Testing Coverage](https://chandrasekarnarayana.github.io/foodspec/06-developer-guide/testing_coverage

For more examples and tutorials, see the [documentation](https://chandrasekarnarayana.github.io/foodspec/).

## Documentation

- **Getting started:** [Installation](https://chandrasekarnarayana.github.io/foodspec/installation/)
- **Quickstart guides:** [Python](https://chandrasekarnarayana.github.io/foodspec/quickstart_python/) • [CLI](https://chandrasekarnarayana.github.io/foodspec/quickstart_cli/) • [Protocols](https://chandrasekarnarayana.github.io/foodspec/quickstart_protocol/)
- **Data & IO:** [CSV import](https://chandrasekarnarayana.github.io/foodspec/csv_to_library/) • [Vendor formats](https://chandrasekarnarayana.github.io/foodspec/vendor_io/) • [Libraries](https://chandrasekarnarayana.github.io/foodspec/libraries/)
- **Preprocessing:** [Complete guide](https://chandrasekarnarayana.github.io/foodspec/preprocessing_guide/)
- **Features & Analysis:** [Feature extraction](https://chandrasekarnarayana.github.io/foodspec/ml_model_vip_scores/) • [Interpretation](https://chandrasekarnarayana.github.io/foodspec/advanced_deep_learning/)
- **Workflows:** [Oil authentication](https://chandrasekarnarayana.github.io/foodspec/protocols_overview/) • [Heating analysis](https://chandrasekarnarayana.github.io/foodspec/aging_workflows/) • [Mixtures](https://chandrasekarnarayana.github.io/foodspec/multimodal_workflows/) • [Calibration](https://chandrasekarnarayana.github.io/foodspec/workflows_harmonization_automated_calibration/)
- **ML & Statistics:** [Methods](https://chandrasekarnarayana.github.io/foodspec/method_comparison/) • [Metrics](https://chandrasekarnarayana.github.io/foodspec/validation_baseline/)
- **Advanced:** [Protocols & automation](https://chandrasekarnarayana.github.io/foodspec/protocols_overview/) • [Registry & plugins](https://chandrasekarnarayana.github.io/foodspec/registry_and_plugins/) • [Deployment](https://chandrasekarnarayana.github.io/foodspec/deployment_artifact_versioning/)
- **Reference:** [Glossary](https://chandrasekarnarayana.github.io/foodspec/glossary/) • [API](https://chandrasekarnarayana.github.io/foodspec/api/) • [Troubleshooting](https://chandrasekarnarayana.github.io/foodspec/troubleshooting_faq/)

## Testing

```bash
pytest --cov          # Run tests with coverage report
ruff check            # Lint checks
mkdocs build          # Build documentation locally
```

## Citation

If you use FoodSpec in your research, please cite the software. See [CITATION.cff](CITATION.cff) for full details.

## Contributing

We welcome contributions. Before submitting, please:
- Follow guidelines in [docs/contributing.md](https://chandrasekarnarayana.github.io/foodspec/contributing/)
- Write clear code with docstrings and examples
- Add tests for new features
- Ensure `pytest`, `ruff`, and `mkdocs build` pass

Open issues and pull requests with concise, clear descriptions.

## Collaborators

- Dr. Jhinuk Gupta, Department of Food and Nutritional Sciences, Sri Sathya Sai Institute of Higher Learning (SSSIHL), Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/dr-jhinuk-gupta-a7070141/)
- Dr. Sai Muthukumar V, Department of Physics, SSSIHL, Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/sai-muthukumar-v-ab78941b/)
- Ms. Amrita Shaw, Department of Food and Nutritional Sciences, SSSIHL, Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/amrita-shaw-246491213/)
- Deepak L. N. Kallepalli, Cognievolve AI Inc., Canada & HCL Technologies Ltd., Bangalore, India — [LinkedIn](https://www.linkedin.com/in/deepak-kallepalli/)

## Author

- Chandrasekar SUBRAMANI NARAYANA, Aix-Marseille University, Marseille, France — [LinkedIn](https://www.linkedin.com/in/snchandrasekar/)

---

FoodSpec aligns spectroscopy, chemometrics, and ML into reproducible, well-documented pipelines for food science. Dive into the [documentation](https://chandrasekarnarayana.github.io/foodspec/) for detailed theory, examples, and workflow guidance.
