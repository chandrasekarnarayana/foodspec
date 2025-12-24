# FoodSpec

<p align="left">
  <img src="docs/assets/foodspec_logo.png" alt="FoodSpec logo" width="200">
</p>

> “Food decides the nature of your mind… Mind is born of the food you take.”  
> — Sri Sathya Sai Baba, *Effect of Food on the Mind*, Summer Showers 1993 – Indian Culture and Spirituality (29 May 1993)

Headless, research-grade Python toolkit for Raman/FTIR (and NIR) spectroscopy in food science. FoodSpec provides a consistent data model, reproducible preprocessing, feature extraction, statistics, ML/chemometrics, and reporting—documented as a textbook-style guide.

**Docs:** https://chandrasekarnarayana.github.io/foodspec/  
**Status:** Public-usage ready (v0.2.0) – tested on Windows/macOS/Linux with example datasets. Vendor binaries: some formats require CSV/HDF5 export.

## Project Intention & Philosophy
- Open, clear, reproducible workflows with strong scientific grounding (physics + chemistry + statistics + ML).
- Real-world usability in labs and QA/QC; interpretable and transparent outputs.
- Accessible to beginners and experts; teaches best practices alongside runnable protocols.
- Vision: a reference toolkit, teaching resource, shared computational language across interdisciplinary teams, and a long-term foundation for reproducible food-science analytics.

## Why FoodSpec?
- Spectroscopy workflows are often fragmented (vendor formats, ad hoc preprocessing, irreproducible scripts).  
- FoodSpec standardizes ingestion (CSV/JCAMP/vendor) into `FoodSpectrumSet`/HDF5 libraries, offers reproducible pipelines, domain workflows (oils, heating, mixtures, QC), and reporting helpers.  
- Documentation serves both as teaching material and protocol-grade guidance.

## Automated analysis
- **CLI + publish:** run a protocol via `foodspec-run-protocol`, then generate narrative/figures with `foodspec-publish` for a fully automated bundle.

## Key capabilities
- **Data & IO:** CSV/TXT/JCAMP, optional vendor loaders (SPC/OPUS), HDF5 spectral libraries; wavenumber-aware validation.
- **Preprocessing:** Baseline (ALS, rubberband, polynomial), smoothing, normalization (vector/area/SNV/MSC), derivatives, cropping, FTIR/Raman helpers, cosmic-ray removal.
- **Features & interpretability:** Peaks/bands/ratios, fingerprint similarity, PCA/PLS scores/loadings, RF importances, peak/ratio summary tables; visualization gallery.
- **Statistics & metrics:** Parametric/nonparametric tests (ANOVA/MANOVA, Tukey, Games–Howell, Mann–Whitney, Kruskal), robustness (bootstrap/permutation), classification/regression metrics, embedding metrics (silhouette, between/within), calibration CIs, Bland–Altman.
- **ML/Chemometrics:** Logistic/SVM (linear/RBF), RF, Gradient Boosting, optional XGBoost/LightGBM, kNN; PLS regression/PLS-DA; optional DL (Conv1D/MLP) with clear cautions; mixture models (NNLS, MCR-ALS).
- **Workflows:** Oil authentication, heating/degradation, mixtures, batch QC/novelty, hyperspectral mapping, calibration/regression—each with plots + metrics + qualitative/quantitative interpretation.
- **Reproducibility & reporting:** CLI/ configs, run metadata, metrics/plots/report.md, model registry, reproducibility checklist, reporting guidelines, troubleshooting/FAQ.
- **Trust & guardrails:** Friendly error messages with auto-fix hints, prediction QC guards (confidence/entropy/margin) that emit `qc_do_not_trust`/`qc_notes` columns in CLI predictions.

## 🎯 Differentiating Capabilities ("Moats")

FoodSpec provides three unique, production-grade toolkits that set it apart in food spectroscopy:

### 1. Matrix Correction
Correct for matrix effects (chips vs. pure oil, emulsions, etc.) with:
- Background subtraction (air/dark/foil references, adaptive baseline via ALS)
- Robust per-matrix scaling (median/MAD, Huber, MCD)
- Domain adaptation via subspace alignment
- Matrix effect magnitude metric for QC

**Key assumptions documented:** Background references measured under identical conditions; matrix types known; ≥10 samples per matrix for adaptation.

```python
fs.apply_matrix_correction(
    method="adaptive_baseline",
    scaling="median_mad",
    domain_adapt=True,
    matrix_column="matrix_type"
)
```

### 2. Heating/Oxidation Trajectory Analysis
Track degradation over time with:
- Oxidation index extraction (PI, TFC, OIT proxy, C=C stretch)
- Trajectory modeling (linear/exponential/sigmoidal fits)
- Degradation stage classifier (RandomForest on index features)
- Shelf-life estimation with confidence intervals

**Key assumptions documented:** Time column numeric; longitudinal data; ≥5 time points; monotonic degradation.

```python
traj = fs.analyze_heating_trajectory(
    time_column="time_hours",
    indices=["pi", "tfc"],
    estimate_shelf_life=True,
    shelf_life_threshold=2.0
)
```

### 3. Calibration Transfer Toolkit
Transfer models between instruments with:
- Direct Standardization (DS) and Piecewise DS (PDS) v2 with robust regression
- Drift detection (mean shift + variance ratio)
- Incremental reference updates (exponential weighting)
- Transfer success dashboard (pre/post RMSE/R²/MAE, leverage/outlier counts)

**Key assumptions documented:** Paired standards; spectral alignment done separately; gradual drift; representative transfer samples.

```python
fs.apply_calibration_transfer(
    source_standards=source_std,
    target_standards=target_std,
    method="pds",
    pds_window_size=11
)
```

**See [docs/moats_overview.md](docs/moats_overview.md) for full details, assumptions, and validation.**

---

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

Single-command reproducibility: place dataset, QC, preprocessing, features, modeling, and outputs in `exp.yml`, then run `foodspec run-exp exp.yml` (or `--dry-run` to validate and view hashes). See `docs/quickstart_cli.md` for the schema and example.
Add `--artifact-path myrun.foodspec` to emit a single-file deployment artifact (model + pipeline hashes + provenance) loadable via `foodspec.load_artifact`.

## Protocol engine quickstart

Run a YAML protocol end-to-end (bundled report, tables, figures):
```bash
foodspec-run-protocol \
  --input data/oils.csv \
  --protocol examples/protocols/EdibleOil_Classification_v1.yml \
  --output-dir runs
```

## Create Demo Libraries

Several example configs reference HDF5 spectral libraries under `libraries/` (e.g., `libraries/oils.h5`). If the folder doesn’t exist yet, generate demo libraries using the built-in example dataset:

```bash
python - <<'PY'
from foodspec.data.loader import load_example_oils
from foodspec.io.exporters import to_hdf5
import os
ds = load_example_oils()
os.makedirs('libraries', exist_ok=True)
to_hdf5(ds, 'libraries/oils.h5')
to_hdf5(ds, 'libraries/qc_ref.h5')
to_hdf5(ds, 'libraries/oils_heating.h5')
print('Created libraries: oils.h5, qc_ref.h5, oils_heating.h5')
PY
```

Mixture demos require pure component references; provide a small pure-set CSV in long format (`sample_id,wavenumber,intensity,label`) and convert it via:

```bash
foodspec csv-to-library pure_long.csv libraries/pure_refs.h5 --format long --modality raman --wavenumber-column wavenumber --sample-id-column sample_id --intensity-column intensity --label-column label
```

See the Smoke Test in [docs/integration_checklist.md](docs/integration_checklist.md) for quick end-to-end verification commands (protocol run with spike toggles, QC, library search, experiment dry-run).

## Prediction service
- CLI: `foodspec-predict` for batch/offline scoring of frozen models.

## Registry & Plugins
- Runs/models can be logged to a registry (`FeatureModelRegistry`). CLI: `foodspec-registry` to list/query.
- Extend FoodSpec via plugins (protocols, vendor loaders, harmonization). CLI: `foodspec-plugin list`. See `examples/plugins/` for starter templates.
  - Documentation: [docs/registry_and_plugins.md](docs/registry_and_plugins.md)

## Related Docs
- Spectral Library Search: [docs/library_search.md](docs/library_search.md)
- Method Comparison (Bland–Altman): [docs/method_comparison.md](docs/method_comparison.md)

## Try these notebooks first
- `examples/notebooks/01_oil_discrimination_basic.ipynb`
- `examples/notebooks/02_oil_vs_chips_matrix_effects.ipynb`
- `examples/notebooks/03_hsi_surface_mapping.ipynb`

## Testing & lint
```bash
pytest --cov      # coverage >90%
ruff check        # lint
mkdocs build      # docs/link check
```

## Citation
If you use FoodSpec, please cite the software (see `CITATION.cff`).

## Contributing
Follow the standards in `docs/dev/developer_notes.md`: clear docstrings with examples, tests for new features, and reproducible workflows. Open an issue/PR with a concise description, and ensure lint/tests/docs pass.  

## Collaborators
- Dr. Jhinuk Gupta, Department of Food and Nutritional Sciences, Sri Sathya Sai Institute of Higher Learning (SSSIHL), Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/dr-jhinuk-gupta-a7070141/)
- Dr. Sai Muthukumar V, Department of Physics, SSSIHL, Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/sai-muthukumar-v-ab78941b/)
- Ms. Amrita Shaw, Department of Food and Nutritional Sciences, SSSIHL, Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/amrita-shaw-246491213/)
- Deepak L. N. Kallepalli, Cognievolve AI Inc., Canada & HCL Technologies Ltd., Bangalore, India — [LinkedIn](https://www.linkedin.com/in/deepak-kallepalli/)

## Author
- Chandrasekar SUBRAMANI NARAYANA, Aix-Marseille University, Marseille, France — [LinkedIn](https://www.linkedin.com/in/snchandrasekar/)

---
FoodSpec aligns spectroscopy, chemometrics, and ML into reproducible, well-documented pipelines for food science. Dive into the docs for detailed theory, examples, and workflow guidance.
