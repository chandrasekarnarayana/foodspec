# FoodSpec

<p align="left">
  <img src="docs/assets/foodspec_logo.png" alt="FoodSpec logo" width="200">
</p>

> "Food decides the nature of your mind… Mind is born of the food you take."  
> — Sri Sathya Sai Baba, *Effect of Food on the Mind*, Summer Showers 1993 – Indian Culture and Spirituality (29 May 1993)

Headless, research-grade Python toolkit for Raman/FTIR/NIR spectroscopy in food science. FoodSpec provides reproducible workflows for preprocessing, feature extraction, statistics, and machine learning, with built-in support for protocol-driven analysis, model management, and automated reporting.

## What problems does FoodSpec solve?

- **Fragmented workflows:** Vendor-specific formats, ad hoc preprocessing, irreproducible scripts.
- **Lack of standards:** No consistent data model across labs or instruments.
- **Manual documentation:** Time-consuming figure and report generation.
- **Reproducibility challenges:** Difficulty sharing, versioning, and archiving analyses.

## What does FoodSpec provide?

**Data & Import**
- Unified data model for Raman, FTIR, and NIR spectroscopy
- CSV/TXT/JCAMP loaders; optional vendor support (SPC, OPUS)
- HDF5 spectral libraries for reference materials and calibration

**Preprocessing**
- Baseline correction, smoothing, normalization, derivatives, cropping
- Cosmic-ray removal and modality-specific helpers
- Wavenumber-aware validation

**Feature Extraction & Interpretation**
- Peak, band, and ratio detection with library-based chemical interpretation
- PCA/PLS scores, fingerprint similarity, feature importance summaries
- Visualization gallery

**Statistics & Quality Control**
- Parametric and nonparametric hypothesis tests
- Classification and regression metrics
- Bootstrap and permutation-based robustness
- Batch QC, novelty detection, drift monitoring

**Machine Learning**
- Classification: logistic, SVM, random forest, gradient boosting
- Regression: linear, partial least squares, random forest
- Optional: XGBoost, LightGBM, Conv1D, and MLP deep learning
- Model registry and versioning

**Domain Workflows**
- Oil authentication and quality assessment
- Heating/oxidation trajectory analysis
- Mixture composition estimation
- Hyperspectral imaging and mapping
- Calibration transfer between instruments

**Reproducibility & Reporting**
- Protocol-driven execution (YAML configuration)
- Automated narrative reports with metrics, tables, and figures
- Run metadata capture and versioning
- Prediction confidence guards
- Reproducibility checklists

## Supported modalities

| Modality | Input | Preprocessing | Workflows |
|----------|-------|---|---|
| **Raman** | Vendor/CSV/HDF5 | Baseline, smoothing, cosmic-ray removal | Authentication, heating, mixtures, QC |
| **FTIR** | Vendor/CSV/HDF5 | Baseline, normalization, cropping | Authentication, heating, QC |
| **NIR** | CSV/HDF5 | Smoothing, derivatives, SNV | Calibration, regression, quality |
| **Hyperspectral** | HDF5 | Per-pixel preprocessing | Mapping, segmentation, classification |

## Installation

```bash
pip install foodspec

# Optional extras
pip install 'foodspec[ml]'      # XGBoost, LightGBM
pip install 'foodspec[deep]'    # Conv1D, MLP deep learning
pip install 'foodspec[dev]'     # Documentation, tests, linting
```

## Quickstart

### Python (5 minutes)

```python
from foodspec.data.loader import load_example_oils
from foodspec.apps.oils import run_oil_authentication_quickstart

fs = load_example_oils()
result = run_oil_authentication_quickstart(fs, label_column="oil_type")
print(result.cv_metrics)
```

### CLI (5 minutes)

```bash
# Preprocess a dataset
foodspec preprocess raw_folder preprocessed.h5 \
  --metadata-csv meta.csv --modality raman

# Run an oil authentication workflow
foodspec oil-auth preprocessed.h5 \
  --label-column oil_type --output-report report.html
```

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
