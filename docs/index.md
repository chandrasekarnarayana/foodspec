# FoodSpec: Open-Source Food Spectroscopy Toolkit

FoodSpec is an open-source toolkit for food authentication and quality control using Raman, FTIR, NIR and HSI spectroscopy. It turns raw spectra into actionable metrics, reports and reproducible workflows.

![FoodSpec logo](assets/foodspec_logo.png)

---

## What is FoodSpec?

FoodSpec automates food authentication and quality control using spectroscopic analysis. It converts raw spectra (light absorption/reflection patterns) into actionable quality metrics: is this oil authentic? Is it degrading? Which adulterants are present?

**In plain terms:** FoodSpec is a "fingerprint scanner" for food. Different oils, fats, and processed foods have distinct spectroscopic signatures. FoodSpec finds those patterns and classifies samples in minutes.

**Scope:** Research, screening, and decision support—not regulatory certification without additional validation.

---

## Who is This For?

| You are... | Your need | Start here |
|-----------|----------|-----------|
| **Food scientist / QA analyst** | Authenticate oils, track thermal stability, screen for adulteration | [Oil Authentication (15 min)](tutorials/intermediate/01-oil-authentication.md) |
| **Spectroscopist** | Automate preprocessing, compare methods, validate results | [Preprocessing Recipes](methods/preprocessing/normalization_smoothing.md) |
| **Physicist / Chemometrician** | Understand algorithms, failure modes, assumptions | [Spectroscopy Theory](theory/spectroscopy_basics.md) |
| **Data scientist / ML engineer** | Train custom models, integrate into pipelines | [API Reference](api/index.md) |
| **DevOps / Software engineer** | Deploy in production, batch processing, automation | [CLI Guide](user-guide/cli.md) |
| **Reviewer / Auditor** | Verify scientific soundness and reproducibility | [Validation Strategies](05-advanced-topics/validation_strategies.md) |

---

## Quick Start (Choose Your Path)

### 🚀 **Path A: CLI (5–10 min)**

**No coding required. Use the command line to analyze CSV files.**

```bash
# 1. Install
pip install foodspec

# 2. Download sample data
wget https://github.com/chandrasekarnarayana/foodspec/raw/main/examples/data/oils.csv

# 3. Run oil authentication workflow
foodspec-run-protocol \
  --input oils.csv \
  --protocol oil_authentication \
  --output-dir runs/demo

# 4. View results
cat runs/demo/*/report.txt
```

**What you get:** Classification report, confusion matrix, metrics, ranked discriminative features.

👉 **Learn more:** [CLI Quickstart](getting-started/quickstart_cli.md)

---

### 🐍 **Path B: Python API (10–15 min)**

**Integrate FoodSpec into Python scripts or Jupyter notebooks.**

```python
from foodspec import SpectralDataset
from foodspec.features.rq import RatioQualityEngine
import pandas as pd

# 1. Load spectra
ds = SpectralDataset.from_csv("oils.csv", wavenumber_col="wavenumber")
print(f"Loaded {len(ds)} spectra")

# 2. Preprocess
ds = ds.preprocess(
    baseline="als",
    normalize="vector",
    smooth=True
)

# 3. Extract ratiometric features
rq = RatioQualityEngine()
ratios = rq.compute(ds)
print(ratios.head())

# 4. Validate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(clf, ratios.features, ratios.labels, cv=5)
print(f"Cross-validated accuracy: {scores.mean():.3f}")
```

👉 **Learn more:** [Python Quickstart](getting-started/quickstart_python.md)

---

## Core Concepts (60 seconds)

### **Data Format**
- Input: CSV, HDF5, or vendor files (Bruker OPUS, Thermo .SPA, Agilent .DPT)
- Format: One spectrum per row; wavenumber/wavelength as columns
- Example: 1000 spectra × 2000 wavenumbers = 2000 intensity values per sample

### **Spectral Preprocessing**
Remove noise and artifacts before analysis:
- **Baseline correction** (ALS, polynomial) — removes instrument drift
- **Normalization** (vector, min-max, reference peak) — makes samples comparable
- **Smoothing** (Savitzky-Golay) — reduces high-frequency noise

### **Feature Extraction**
Convert spectra into interpretable metrics:
- **Ratiometric features** — ratio of two peak areas (e.g., C=O / C=C bands) → quick screening
- **Peak positions & widths** — detect molecular changes
- **Dimensionality reduction** (PCA, PLS) — compress high-dimensional data

### **Classification & Validation**
- Train ML models (Logistic Regression, Random Forest, XGBoost)
- Validate with nested cross-validation and batch-aware splitting
- Report: confusion matrix, ROC curve, feature importance

### **Reproducibility**
Every run creates a timestamped bundle with:
- Preprocessing config
- Model weights
- Metrics and diagnostics
- Methods text for publication

---

## Core Workflows (Curated Examples)

| Workflow | What it does | Skill level | Time |
|----------|-------------|------------|------|
| **[Oil Authentication](workflows/authentication/oil_authentication.md)** | Classify edible oils; detect common adulterants | Beginner | 20 min |
| **[Heating Quality Monitoring](workflows/quality-monitoring/heating_quality_monitoring.md)** | Track frying oil degradation; estimate shelf life | Intermediate | 30 min |
| **[Mixture Analysis](workflows/quantification/mixture_analysis.md)** | Quantify composition of oil blends | Intermediate | 30 min |
| **[Batch QC](workflows/batch_quality_control.md)** | Screen incoming ingredients for consistency | Beginner | 15 min |
| **[Hyperspectral Mapping](workflows/spatial/hyperspectral_mapping.md)** | Analyze spatial composition of food surfaces | Advanced | 45 min |

👉 See all: [Workflows](workflows/authentication/oil_authentication.md)

---

## When NOT to Use FoodSpec

**✅ DO use FoodSpec for:**
- Rapid screening and decision support
- Hypothesis generation and pattern discovery
- Comparison of batches or production runs
- Training and method development

**❌ DO NOT use FoodSpec for:**
- Regulatory certification (ISO, FDA, etc.)
- Legal claims of purity or authenticity
- Autonomous systems without human review
- Compounds below detection limit
- Definitive pathogen/toxin detection

**👉 Full details:** [Non-Goals & Limitations](non_goals_and_limitations.md)

---

## Citing FoodSpec

If you use FoodSpec in research or publication, please cite:

**Bibtex:**
```bibtex
@software{foodspec2024,
  author = {Chandrasekar, S. N. and others},
  title = {FoodSpec: Open-source toolkit for food spectroscopy},
  url = {https://github.com/chandrasekarnarayana/foodspec},
  year = {2024},
  note = {v1.0.0}
}
```

**APA:**
Chandrasekar, S. N., et al. (2024). *FoodSpec: Open-source toolkit for food spectroscopy* (v1.0.0). https://github.com/chandrasekarnarayana/foodspec

👉 **Full citation guide:** [Citing FoodSpec](reference/citing.md)

---

## Documentation Structure

| Section | What you'll find |
|---------|-----------------|
| **[Start Here](getting-started/installation.md)** | Installation, quickstarts, FAQ |
| **[Tutorials](tutorials/index.md)** | Step-by-step examples (oil auth, HSI, etc.) |
| **[Methods & Validation](methods/validation/index.md)** | Preprocessing, validation, troubleshooting |
| **[User Guide](user-guide/index.md)** | CLI, protocols, data formats, automation |
| **[Theory & Background](theory/spectroscopy_basics.md)** | Spectroscopy, chemometrics, algorithms |
| **[API Reference](api/index.md)** | Python function documentation |
| **[Glossary](reference/glossary.md)** | Technical terms defined |

---

## Get Help

- **Installation issues or errors?** → [Troubleshooting](help/troubleshooting.md) – Solutions for common technical problems
- **Common questions?** → [FAQ](help/faq.md) – Baseline methods, sample size, citations, and more
- **New to spectroscopy?** → [Spectroscopy Basics](theory/spectroscopy_basics.md)
- **Have a question?** → [GitHub Discussions](https://github.com/chandrasekarnarayana/foodspec/discussions)
- **Found a bug?** → [GitHub Issues](https://github.com/chandrasekarnarayana/foodspec/issues)

---

## Keywords

- food spectroscopy
- Raman
- FTIR
- NIR
- HSI
- food authentication
- quality control
- chemometrics
- machine learning

## Acknowledgments

FoodSpec is developed by an international team:

**Core team:**
- Dr. Chandrasekar Subramani Narayana (Aix-Marseille University)
- Dr. Jhinuk Gupta (Sri Sathya Sai Institute of Higher Learning)
- Dr. Sai Muthukumar V (Sri Sathya Sai Institute of Higher Learning)
- Ms. Amrita Shaw (Sri Sathya Sai Institute of Higher Learning)
- Deepak L. N. Kallepalli (Cognievolve AI Inc., HCL Technologies)

**Citation:** If you publish using FoodSpec, please cite both the code and this collaborative work.

---

**Version:** 1.0.0 | **Last updated:** December 2024
