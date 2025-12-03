# FoodSpec

<p align="left">
  <img src="docs/assets/foodspec_logo.png" alt="FoodSpec logo" width="200">
</p>

Headless, research-grade Python toolkit for Raman/FTIR (and NIR) spectroscopy in food science. FoodSpec provides a consistent data model, reproducible preprocessing, feature extraction, statistics, ML/chemometrics, and reporting—documented as a textbook-style guide.

**Docs:** https://chandrasekarnarayana.github.io/foodspec/  
**Status:** Tests pass; coverage >90%; lint (ruff) enabled.

## Why FoodSpec?
- Spectroscopy workflows are often fragmented (vendor formats, ad hoc preprocessing, irreproducible scripts).  
- FoodSpec standardizes ingestion (CSV/JCAMP/vendor) into `FoodSpectrumSet`/HDF5 libraries, offers reproducible pipelines, domain workflows (oils, heating, mixtures, QC), and reporting helpers.  
- Documentation serves both as teaching material and protocol-grade guidance.

## Key capabilities
- **Data & IO:** CSV/TXT/JCAMP, optional vendor loaders (SPC/OPUS), HDF5 spectral libraries; wavenumber-aware validation.
- **Preprocessing:** Baseline (ALS, rubberband, polynomial), smoothing, normalization (vector/area/SNV/MSC), derivatives, cropping, FTIR/Raman helpers, cosmic-ray removal.
- **Features & interpretability:** Peaks/bands/ratios, fingerprint similarity, PCA/PLS scores/loadings, RF importances, peak/ratio summary tables; visualization gallery.
- **Statistics & metrics:** Parametric/nonparametric tests (ANOVA/MANOVA, Tukey, Games–Howell, Mann–Whitney, Kruskal), robustness (bootstrap/permutation), classification/regression metrics, embedding metrics (silhouette, between/within), calibration CIs, Bland–Altman.
- **ML/Chemometrics:** Logistic/SVM (linear/RBF), RF, Gradient Boosting, optional XGBoost/LightGBM, kNN; PLS regression/PLS-DA; optional DL (Conv1D/MLP) with clear cautions; mixture models (NNLS, MCR-ALS).
- **Workflows:** Oil authentication, heating/degradation, mixtures, batch QC/novelty, hyperspectral mapping, calibration/regression—each with plots + metrics + qualitative/quantitative interpretation.
- **Reproducibility & reporting:** CLI/ configs, run metadata, metrics/plots/report.md, model registry, reproducibility checklist, MethodsX-style protocol, troubleshooting/FAQ.

## Installation
```bash
pip install foodspec
# Optional extras
pip install 'foodspec[ml]'      # xgboost/lightgbm
pip install 'foodspec[deep]'    # Conv1D/MLP deep models
pip install 'foodspec[dev]'     # docs/tests/lint (mkdocs, ruff, pytest)
```

## Quickstart (Python)
```python
from foodspec.data.loader import load_example_oils
from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.metrics import compute_classification_metrics

fs = load_example_oils()
res = run_oil_authentication_quickstart(fs, label_column="oil_type")
print(res.cv_metrics)
```
For calibration/regression, see `docs/workflows/calibration_regression_example.md` and the PLS example in `docs/ml/classification_regression.md`.

## Quickstart (CLI)
```bash
foodspec preprocess raw_folder preprocessed.h5 --metadata-csv meta.csv --modality raman
foodspec oil-auth preprocessed.h5 --label-column oil_type --output-report report.html
```
See `docs/cli.md` and workflow pages for full CLI configs and outputs (metrics.json, plots, report.md).

## Testing & lint
```bash
pytest --cov      # coverage >90%
ruff check        # lint
mkdocs build      # docs/link check
```

## Citation
If you use FoodSpec, please cite the software (see `CITATION.cff`) and the forthcoming MethodsX protocol (placeholder until DOI is available).

## Contributing
Follow the standards in `docs/dev/developer_notes.md`: clear docstrings with examples, tests for new features, and reproducible workflows. Open an issue/PR with a concise description, and ensure lint/tests/docs pass.  

---
FoodSpec aligns spectroscopy, chemometrics, and ML into reproducible, well-documented pipelines for food science. Dive into the docs for detailed theory, examples, and workflow guidance.
