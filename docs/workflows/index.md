# Workflows: Reproducible Analysis Patterns

Reproducible end‑to‑end workflows for authentication, degradation monitoring, mixture analysis, harmonization and hyperspectral mapping.

---

## 🗺️ Workflow Categories

### **Authentication & Identification**
Determine what a sample is (classification).

| Workflow | Problem | Time | Difficulty |
|----------|---------|------|------------|
| [Oil Authentication](authentication/oil_authentication.md) | "What oil is this?" / "Is it adulterated?" | 30 min | Beginner |
| [Matrix Effects](domain_templates.md) | Compare markers across matrices (oils vs chips) | 40 min | Applied |

**When to use:** Verify authenticity, detect fraud, classify unknowns into known categories.

---

### **Degradation & Thermal Monitoring**
Track chemical changes over time, temperature, or storage.

| Workflow | Problem | Time | Difficulty |
|----------|---------|------|------------|
| [Heating & Quality Monitoring](heating_quality_monitoring.md) | Track oxidation/degradation during frying | 35 min | Beginner |
| [Aging Workflows](aging_workflows.md) | Monitor shelf-life and storage stability | 40 min | Applied |
| [Batch Quality Control](batch_quality_control.md) | Detect drift, outliers, and batch-to-batch variation | 45 min | Applied |

**When to use:** Monitor frying cycles, predict shelf-life, detect off-spec batches, study degradation kinetics.

---

### **Adulteration & Mixture Analysis**
Quantify components in blends or detect contamination.

| Workflow | Problem | Time | Difficulty |
|----------|---------|------|------------|
| [Mixture Analysis](mixture_analysis.md) | Quantify adulteration levels (e.g., 10% seed oil in olive) | 40 min | Applied |
| [Calibration & Regression](quantification/calibration_regression_example.md) | Build calibration curves for quantitative prediction | 50 min | Advanced |

**When to use:** Quantify adulterants, build concentration models, detect contamination thresholds.

---

### **Harmonization & Instrument Effects**
Handle multi-instrument data or transfer models.

| Workflow | Problem | Time | Difficulty |
|----------|---------|------|------------|
| [Harmonization & Automated Calibration](harmonization_automated_calibration.md) | Transfer models between instruments, correct batch effects | 60 min | Advanced |
| [Standard Templates](standard_templates.md) | Create reusable workflow templates for common tasks | 45 min | Advanced |

**When to use:** Combine data from multiple instruments, transfer models to new sites, standardize QA protocols.

---

### **Spatial & Hyperspectral Analysis**
Map chemical composition across surfaces.

| Workflow | Problem | Time | Difficulty |
|----------|---------|------|------------|
| [Hyperspectral Mapping](spatial/hyperspectral_mapping.md) | Map contaminants, coatings, or ROIs on surfaces | 50 min | Advanced |

**When to use:** Visualize spatial distribution, segment regions of interest, analyze surface coatings.

---

### **Workflow Design & Reporting**
Meta-workflow for creating new analysis pipelines.

| Workflow | Problem | Time | Difficulty |
|----------|---------|------|------------|
| [Workflow Design & Reporting](workflow_design_and_reporting.md) | Design custom workflows with proper documentation | 60 min | Advanced |

**When to use:** Build new domain-specific workflows, document analysis procedures, ensure reproducibility.

---

## 📋 Workflow Structure

Every FoodSpec workflow follows a consistent template:

### 1. Standard Header
- **Purpose:** One-sentence problem statement
- **When to Use:** Specific scenarios where this workflow applies
- **Inputs:** Required data format and metadata columns
- **Outputs:** Expected results (plots, tables, metrics)
- **Assumptions:** What the workflow assumes about your data

### 2. Minimal Reproducible Example (MRE)
- Synthetic data generator **or** bundled example dataset
- Copy-paste code that runs without external files
- Complete workflow from load → preprocess → model → results

### 3. Validation & Sanity Checks
- **Success indicators:** What plots/metrics look like when working correctly
- **Failure indicators:** Red flags that something is wrong
- **Quality thresholds:** Minimum acceptable performance

### 4. Parameters You Must Justify
- Critical parameters (baseline λ, smoothing window, CV folds)
- When to adjust from defaults
- How to document parameter choices

---

## 🚀 Quick Start Guide

### New to FoodSpec?
1. Start with [Oil Authentication](authentication/oil_authentication.md) (simplest workflow)
2. Try [Heating & Quality Monitoring](quality-monitoring/heating_quality_monitoring.md) (time-series analysis)
3. Explore [Workflow Design & Reporting](workflow_design_and_reporting.md) (custom workflows)

### Have your own data?
1. Check the **Inputs** section of relevant workflow
2. Ensure your data matches the format (CSV or HDF5 with required metadata)
3. Run the MRE with your data path substituted
4. Review **Validation & Sanity Checks** to verify results

### Building a new workflow?
1. Read [Workflow Design & Reporting](workflow_design_and_reporting.md)
2. Use [Standard Templates](standard_templates.md) as starting point
3. Follow the standard structure (Header → MRE → Validation → Parameters)

---

## 🔍 Choosing the Right Workflow

### Decision Tree

```plaintext
What's your goal?
├─ Identify/classify samples?
│  └─ Oil Authentication
├─ Track degradation over time?
│  ├─ Heating cycles? → Heating & Quality Monitoring
│  └─ Storage/shelf-life? → Aging Workflows
├─ Quantify adulterants?
│  ├─ Discrete levels? → Mixture Analysis
│  └─ Continuous concentration? → Calibration & Regression
├─ Handle multiple instruments?
│  └─ Harmonization & Automated Calibration
├─ Map surfaces spatially?
│  └─ Hyperspectral Mapping
└─ Build custom workflow?
   └─ Workflow Design & Reporting
```

---

## 📊 Workflow Comparison

| Feature | Authentication | Degradation | Adulteration | Harmonization |
|---------|----------------|-------------|--------------|---------------|
| **Output Type** | Classification | Regression/Trends | Quantification | Model Transfer |
| **Metadata Required** | Labels | Time/Temperature | Concentration | Instrument ID |
| **Typical Duration** | 30–40 min | 35–45 min | 40–50 min | 60+ min |
| **Model Type** | RF, SVM, PLS-DA | Linear, ANCOVA | NNLS, MCR-ALS | DS, PDS, ComBat |
| **Validation** | CV + Confusion Matrix | R², RMSE, Trends | R², Calibration Curve | Transfer Accuracy |

---

## ⚙️ Common Parameters Across Workflows

### Preprocessing (Universal)
- **Baseline correction:** ALS (λ=1e4, p=0.01) — Remove background curvature
- **Smoothing:** Savitzky-Golay (window=21, polyorder=3) — Reduce noise
- **Normalization:** SNV or L2 — Scale spectra to unit norm
- **Cropping:** Spectral region (e.g., 600–1800 cm⁻¹) — Focus on informative peaks

### Modeling (Task-Specific)
- **Classification:** Random Forest (n_trees=100, max_depth=None)
- **Regression:** Linear or Ridge (α=1.0)
- **Validation:** 5-fold stratified CV (for classification), 5-fold CV (for regression)

### Reporting (Universal)
- **Plots:** Confusion matrix, PCA scores, ratio trends, calibration curves
- **Tables:** Metrics (accuracy, R², RMSE), feature importance, ANOVA results
- **Narrative:** report.md summarizing findings

**See individual workflows for parameter justification guidance.**

---

## 🧪 Example Data Requirements

| Workflow | Min Samples | Metadata Columns | Typical Wavenumber Range |
|----------|-------------|------------------|--------------------------|
| Oil Authentication | 50–100 | `oil_type`, `batch` (optional) | 600–1800 cm⁻¹ |
| Heating Monitoring | 30–50 | `heating_time`, `oil_type` (optional) | 600–1800 cm⁻¹ |
| Mixture Analysis | 40–80 | `concentration`, `mixture_type` | 600–1800 cm⁻¹ |
| Batch QC | 100+ | `batch`, `date`, `instrument` | 600–1800 cm⁻¹ |
| Harmonization | 50+ per instrument | `instrument_id`, `batch` | Full range |

---

## 📚 Related Documentation

- **[Tutorials](../tutorials/index.md)** — Step-by-step learning paths

## Keywords

- oil authentication
- heating quality
- mixture analysis
- harmonization
- hyperspectral mapping
- **[Cookbook](../workflows/index.md)** — Recipe-style how-to guides
- **[User Guide](../user-guide/index.md)** — CLI and automation
- **[Theory](../theory/index.md)** — Scientific foundations
- **[API Reference](../08-api/index.md)** — Function/class documentation

---

## 🐛 Troubleshooting

Common issues across workflows:

1. **"Model accuracy too low"** → Check preprocessing parameters, SNR, class balance
2. **"Trends not significant"** → Increase sample size, check metadata alignment
3. **"Harmonization fails"** → Verify instrument IDs, check spectral alignment
4. **"Plots don't render"** → Check matplotlib backend, file paths

See [Troubleshooting Guide](../troubleshooting/troubleshooting_faq.md) for detailed solutions.

---

## 💡 Best Practices

1. **Always start with MRE** — Verify workflow works with synthetic data first
2. **Document parameter choices** — Justify baseline λ, smoothing window, CV folds
3. **Check validation metrics** — Don't trust the model until you've validated it
4. **Generate reproducible reports** — Use FoodSpec's auto-reporting tools
5. **Version control workflows** — Store YAML protocols in Git alongside data

---

## 🎯 Success Criteria

After completing a workflow, you should have:

✅ **Plots:** Confusion matrix, PCA scores, or trend plots (depending on workflow)  
✅ **Tables:** Metrics (accuracy, R², RMSE), feature importance, or ANOVA results  
✅ **Narrative:** report.md summarizing findings and interpretation  
✅ **Reproducibility:** YAML protocol or Python script that can be re-run  
✅ **Validation:** Cross-validation metrics or test set results  

---

## 🔗 Quick Links

- **Beginner-Friendly:** [Oil Authentication](authentication/oil_authentication.md), [Heating Monitoring](quality-monitoring/heating_quality_monitoring.md)
- **Most Common:** [Batch QC](quality-monitoring/batch_quality_control.md), [Mixture Analysis](quantification/mixture_analysis.md)
- **Advanced:** [Harmonization](harmonization/harmonization_automated_calibration.md), [Hyperspectral](spatial/hyperspectral_mapping.md)
- **Meta:** [Workflow Design](workflow_design_and_reporting.md), [Templates](harmonization/standard_templates.md)

Happy analyzing! 🔬
