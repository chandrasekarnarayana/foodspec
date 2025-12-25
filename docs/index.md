# FoodSpec Documentation

![FoodSpec logo](assets/foodspec_logo.png)

> "Food decides the nature of your mind… Mind is born of the food you take."  
> — Sri Sathya Sai Baba, *Effect of Food on the Mind*, Summer Showers 1993 – Indian Culture and Spirituality (29 May 1993)

---

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Scientists, engineers, lab technicians, and students working with food spectroscopy (Raman/FTIR/NIR/HSI).  
**What problem does this solve?** Automating food authentication, quality control, and chemical analysis workflows with reproducible, validated pipelines.  
**When to use this?** When you need to authenticate oils, detect adulteration, track thermal degradation, or analyze food composition using spectroscopy.  
**Why it matters?** Food fraud costs billions annually. FoodSpec provides open-source, scientifically validated tools for quality assurance.  
**Time to complete:** 15 min (quickstart) to several hours (custom workflows)  
**Prerequisites:** Python 3.10+, spectral data (CSV/HDF5), basic command-line knowledge

---

## What is FoodSpec? (4-Layer Explanation)

### Layer 1: Plain English (Layman)
FoodSpec is software that helps identify fake or low-quality food using light-based measurements. Think of it like a "fingerprint scanner" for food—different foods reflect light differently, and FoodSpec finds those patterns automatically.

**Example:** Shine a laser on olive oil vs. sunflower oil. They look the same to your eyes, but their "light fingerprints" are different. FoodSpec detects this and tells you which is which.

### Layer 2: Domain Expert (Food Scientist / Spectroscopist)
FoodSpec is a protocol-driven toolkit for Raman/FTIR spectroscopy and hyperspectral imaging in food science. It automates:
- Edible oil authentication (olive, palm, sunflower, coconut)
- Thermal stability tracking (frying oil degradation)
- Matrix effect analysis (oil–chips interactions)
- Adulterant detection (mixing, oxidation, heating damage)

**Core workflows:** Preprocessing → harmonization → RQ (Ratio-Quality) analysis → validation → reporting.

### Layer 3: Rigorous Theory (Physicist / Chemometrician)
FoodSpec implements vibrational spectroscopy chemometrics:
- **Preprocessing:** Asymmetric Least Squares (ALS) baseline correction, Savitzky-Golay smoothing, reference peak normalization
- **Feature extraction:** Ratiometric features (carbonyl/unsaturation bands 1600-1800 cm⁻¹), peak area integration
- **Classification:** Logistic regression with L1 regularization, Random Forests, XGBoost (batch-aware cross-validation)
- **Validation:** Nested CV, permutation tests, MOATS (Model Optimized by Accumulated Threshold Selection)

**Assumptions:** Linear detector response, negligible fluorescence, stable laser power. **When results cannot be trusted:** SNR < 10, class imbalance > 10:1, batch confounding with labels.

### Layer 4: Developer / API (Software Engineer)
\`\`\`python
from foodspec import SpectralDataset
from foodspec.apps.oils import run_oil_authentication_workflow

# Load Raman spectra
ds = SpectralDataset.from_csv("oils.csv")

# Run protocol-driven workflow
results = run_oil_authentication_workflow(ds, validation_mode="batch_aware")

# Access metrics
print(results.balanced_accuracy)  # 0.945
print(results.confusion_matrix)
\`\`\`

**CLI equivalent:**
\`\`\`bash
foodspec oil-auth --input oils.csv --output-dir runs/
\`\`\`

**Extensibility:** Protocol YAML + plugin system for custom steps, vendor loaders, and custom metrics.

---

## ⚠️ Before Using FoodSpec: Important Scope & Limitations

**FoodSpec is designed for research, screening, and decision support—not as a replacement for regulatory reference methods or autonomous quality control without human review.**

**Please read:** [Non-Goals & Limitations](non_goals_and_limitations.md) — Understand what FoodSpec can and cannot do, when results may be misleading, and when to seek expert guidance.

**Key points:**
- ✅ Use FoodSpec for rapid screening, hypothesis generation, and pattern discovery
- ✅ Combine with reference methods (GC-MS, HPLC, etc.) for critical decisions
- ✅ Always use ≥3 replicates and validate on independent test data
- ❌ Do NOT use for absolute purity claims or regulatory certifications without verification
- ❌ Do NOT trust results >95% accuracy without investigation (likely overfitting or data leakage)
- ❌ Do NOT skip batch effect management in cross-instrument deployment

---

## Who is This For? (7 Audiences, 7 Entry Paths)

### 1. Absolute Beginner (Layman)
**You are:** Someone with zero spectroscopy experience who wants to test food samples.  
**Start here:** [15-Minute Quickstart](01-getting-started/quickstart_15min.md) → Run first analysis in 15 minutes.  
**Goal:** Verify FoodSpec works on your system; understand basic workflow.

### 2. Food Scientist / QA Analyst
**You are:** Running routine oil authentication or quality control in a lab.  
**Start here:** [Installation](01-getting-started/installation.md) → [Oil Discrimination Tutorial](02-tutorials/oil_discrimination_basic.md)  
**Goal:** Learn which workflows apply to your samples; interpret confusion matrices and discriminative ratios.

### 3. Spectroscopist (Raman/FTIR Expert)
**You are:** Experienced with spectroscopy but new to automated workflows.  
**Start here:** [Preprocessing Recipes](03-cookbook/cookbook_preprocessing.md) → [Validation Recipes](03-cookbook/cookbook_validation.md)  
**Goal:** Understand preprocessing choices (baseline correction, normalization); customize validation strategies.

### 4. Physicist / Chemometrician
**You are:** Developing new algorithms or validating existing methods scientifically.  
**Start here:** [Theory & Background](07-theory-and-background/spectroscopy_basics.md) → [RQ Engine Theory](07-theory-and-background/rq_engine_theory.md)  
**Goal:** Understand assumptions, failure modes, and mathematical foundations.

### 5. Data Scientist / ML Engineer
**You are:** Integrating FoodSpec into ML pipelines or custom applications.  
**Start here:** [API Reference](api/index.md) → [Model Lifecycle](05-advanced-topics/model_lifecycle_and_prediction.md)  
**Goal:** Use FoodSpec as a Python library; train/deploy custom models.

### 6. Software Engineer / DevOps
**You are:** Deploying FoodSpec in production or CI/CD pipelines.  
**Start here:** [CLI Guide](04-user-guide/cli_guide.md) → [Automated Analysis](04-user-guide/automation.md)  
**Goal:** Batch processing, Docker deployment, protocol versioning.

### 7. Reviewer / Auditor (Regulatory / Academic)
**You are:** Verifying FoodSpec for compliance, publication, or certification.  
**Start here:** [Validation Strategies](05-advanced-topics/validation_strategies.md) → [Docs Audit Report](archive/DOCS_AUDIT_REPORT.md)  
**Goal:** Review assumptions, limits, failure modes, and reproducibility.

---

## Quick Navigation (by Task)

**I want to...**

- **Run my first analysis** → [15-Minute Quickstart](01-getting-started/quickstart_15min.md)
- **Authenticate edible oils** → [Oil Discrimination Tutorial](02-tutorials/oil_discrimination_basic.md)
- **Track oil degradation during frying** → [Thermal Stability Tutorial](02-tutorials/thermal_stability_tracking.md)
- **Analyze hyperspectral imaging (HSI) data** → [HSI Surface Mapping](02-tutorials/hsi_surface_mapping.md)
- **Create a custom protocol** → [Protocols & YAML](04-user-guide/protocols_and_yaml.md)
- **Understand preprocessing options** → [Preprocessing Recipes](03-cookbook/cookbook_preprocessing.md)
- **Validate my results scientifically** → [Validation Strategies](05-advanced-topics/validation_strategies.md)
- **Extend FoodSpec with plugins** → [Writing Plugins](06-developer-guide/writing_plugins.md)
- **Troubleshoot errors** → [Troubleshooting Cookbook](03-cookbook/cookbook_troubleshooting.md)
- **Understand the theory** → [Spectroscopy Basics](07-theory-and-background/spectroscopy_basics.md)

---

## Collaborators

FoodSpec is developed by an international team of food scientists, physicists, and engineers:

- Dr. Jhinuk Gupta, Department of Food and Nutritional Sciences, Sri Sathya Sai Institute of Higher Learning (SSSIHL), Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/dr-jhinuk-gupta-a7070141/)
- Dr. Sai Muthukumar V, Department of Physics, SSSIHL, Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/sai-muthukumar-v-ab78941b/)
- Ms. Amrita Shaw, Department of Food and Nutritional Sciences, SSSIHL, Andhra Pradesh, India — [LinkedIn](https://www.linkedin.com/in/amrita-shaw-246491213/)
- Deepak L. N. Kallepalli, Cognievolve AI Inc., Canada & HCL Technologies Ltd., Bangalore, India — [LinkedIn](https://www.linkedin.com/in/deepak-kallepalli/)

---

## Author

- Chandrasekar SUBRAMANI NARAYANA, Aix-Marseille University, Marseille, France — [LinkedIn](https://www.linkedin.com/in/snchandrasekar/)

---

## What's Next?

**New Users:**  
→ [15-Minute Quickstart](01-getting-started/quickstart_15min.md) to run your first analysis

**Food Scientists:**  
→ [Oil Discrimination Tutorial](02-tutorials/oil_discrimination_basic.md) for guided workflow

**Developers:**  
→ [API Reference](api/index.md) to integrate FoodSpec into your code

**Researchers:**  
→ [Validation Strategies](05-advanced-topics/validation_strategies.md) for scientific rigor

---

**Need help getting started?** See [FAQ (Basic)](01-getting-started/faq_basic.md) or open a [discussion](https://github.com/chandrasekarnarayana/foodspec/discussions).
