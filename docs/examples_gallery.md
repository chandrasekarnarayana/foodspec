# Examples Gallery

Quick-start recipes for common food spectroscopy tasks. Each recipe shows the problem, minimal working code, expected output, and links to full documentation.

---

## 🎯 Recipe Cards

### 1. Authenticate Cooking Oils

**Problem:** Detect fake or adulterated oils using Raman spectroscopy.

```python
from foodspec import FoodSpec

# Load oil spectra and run authentication
fs = FoodSpec("oils.csv", modality="raman")
result = fs.classify(
    label_column="oil_type",
    model="pls-da",
    cv_folds=5
)

# View results
print(f"Accuracy: {result.accuracy:.2%}")
result.plot_confusion_matrix()
```

**Output:** Confusion matrix showing classification accuracy per oil type; balanced accuracy ~95% on validation set.

**Learn more:** [Oil Authentication Tutorial](tutorials/intermediate/01-oil-authentication.md) • [Oil Workflow](workflows/authentication/oil_authentication.md)

---

### 2. Remove Baseline Drift

**Problem:** Spectra have curved baselines from fluorescence or scattering.

```python
from foodspec.preprocessing import baseline_als

# Load noisy spectra
fs = FoodSpec("raw_spectra.csv")

# Apply asymmetric least squares baseline correction
fs_corrected = fs.apply_baseline(
    method="als",
    lam=1e6,     # Smoothness
    p=0.01       # Asymmetry (low = ignore peaks)
)

fs_corrected.plot(title="Baseline Corrected")
```

**Output:** Flat baselines with preserved peak shapes; ready for quantitative analysis.

**Learn more:** [Baseline Correction Guide](methods/preprocessing/baseline_correction.md) • [Preprocessing Cookbook](methods/preprocessing/normalization_smoothing.md)

---

### 3. Track Oxidation Over Time

**Problem:** Monitor oil degradation during heating/storage to predict shelf life.

```python
from foodspec import FoodSpec

# Load time-series spectra
fs = FoodSpec("heating_study.csv", modality="raman")

# Analyze oxidation trajectory
result = fs.analyze_heating_trajectory(
    time_column="time_hours",
    estimate_shelf_life=True,
    shelf_life_threshold=2.0  # Peroxide index threshold
)

# View shelf life estimate
print(f"Shelf life: {result.shelf_life_estimate} hours")
print(f"95% CI: {result.confidence_interval}")
```

**Output:** Shelf-life prediction with confidence intervals; trajectory plot showing oxidation indices vs. time.

**Learn more:** [Heating Quality Workflow](workflows/quality-monitoring/heating_quality_monitoring.md) • [MOATS Overview](theory/moats_overview.md)

---

### 4. Smooth Noisy Spectra

**Problem:** Raw spectra have high-frequency noise obscuring true signal.

```python
from foodspec import FoodSpec

fs = FoodSpec("noisy_spectra.csv")

# Apply Savitzky-Golay smoothing
fs_smooth = fs.smooth(
    method="savgol",
    window_length=11,
    polyorder=2
)

# Compare before/after
fs.plot(label="Raw", alpha=0.5)
fs_smooth.plot(label="Smoothed", linewidth=2)
```

**Output:** Smoothed spectra preserving peak positions and relative intensities while reducing noise.

**Learn more:** [Normalization & Smoothing](methods/preprocessing/normalization_smoothing.md) • [Preprocessing Guide](methods/preprocessing/normalization_smoothing.md)

---

### 5. Detect Matrix Effects (Domain Shift)

**Problem:** Model trained on oils fails on chips—different food matrices cause spectral shifts.

```python
from foodspec import FoodSpec

# Load mixed-matrix dataset
fs = FoodSpec("oils_and_chips.csv")

# Calculate divergence between matrices
divergence = fs.compute_domain_divergence(
    source_samples=fs.metadata\["matrix"\] == "oil",
    target_samples=fs.metadata\["matrix"\] == "chips"
)

print(f"KL Divergence: {divergence.kl_divergence:.3f}")
print(f"Shift magnitude: {divergence.shift_magnitude:.2f}")
```

**Output:** Quantified domain shift metrics; wavenumber-specific divergence plot showing which peaks differ between matrices.

**Learn more:** [Matrix Effects Tutorial](tutorials/intermediate/02-matrix-effects.md) • [Harmonization Theory](theory/harmonization_theory.md)

---

### 6. Build Reproducible Pipeline

**Problem:** Need versioned, auditable analysis for regulatory submission or publication.

```python
import yaml
from foodspec import FoodSpecProtocol

# Define protocol in YAML
protocol = """
name: oil_authentication_v1
data:
  input_file: oils.csv
  labels_column: oil_type
preprocessing:
  - type: baseline_als
    params: {lam: 1e6, p: 0.01}
  - type: normalize
    params: {method: "snv"}
model:
  type: pls-da
  params: {n_components: 5}
validation:
  cv_strategy: stratified_kfold
  n_splits: 5
"""

# Run protocol and generate report
result = FoodSpecProtocol.from_yaml(protocol).run()
result.export_bundle(path="results/", include_metadata=True)
```

**Output:** Complete output bundle: figures, tables, metadata.json (reproducible record), and auto-generated report.md.

**Learn more:** [Reproducible Pipelines Tutorial](tutorials/advanced/01-reproducible-pipelines.md) • [Protocols & YAML Guide](user-guide/protocols_and_yaml.md)

---

### 7. Normalize for Instrument Drift

**Problem:** Spectra collected on different days have intensity variations from lamp aging.

```python
from foodspec import FoodSpec

fs = FoodSpec("multi_day_spectra.csv")

# Apply Standard Normal Variate (SNV) normalization
fs_norm = fs.normalize(method="snv")

# Or use Min-Max scaling per spectrum
fs_minmax = fs.normalize(method="minmax")

# Check consistency
print(f"Pre-norm std: {fs.x.std(axis=1).mean():.3f}")
print(f"Post-norm std: {fs_norm.x.std(axis=1).mean():.3f}")
```

**Output:** Normalized spectra with consistent intensity ranges; reduced batch effects in PCA scores plot.

**Learn more:** [Normalization & Smoothing](methods/preprocessing/normalization_smoothing.md) • [Preprocessing Cookbook](methods/preprocessing/normalization_smoothing.md)

---

### 8. Identify Key Discriminative Markers

**Problem:** Need to know which spectral regions distinguish product classes (for QC panel design).

```python
from foodspec import FoodSpec

fs = FoodSpec("oils.csv", modality="raman")

# Run classification and extract feature importance
result = fs.classify(
    label_column="oil_type",
    model="random_forest",
    extract_importance=True
)

# View top discriminative features
top_features = result.feature_importance.nlargest(10)
print(top_features)

# Plot discriminative regions
result.plot_feature_importance(top_n=15)
```

**Output:** Ranked list of discriminative wavenumbers/features; barplot showing which spectral regions separate classes.

**Learn more:** [RQ Questions Cookbook](theory/rq_engine_detailed.md) • [Chemometrics Guide](methods/chemometrics/models_and_best_practices.md)

---

### 9. Run Batch-Aware Cross-Validation

**Problem:** Samples from same batch are correlated—naive CV overestimates performance.

```python
from foodspec import FoodSpec

fs = FoodSpec("batch_data.csv")

# Use GroupKFold to prevent data leakage
result = fs.classify(
    label_column="quality",
    model="pls-da",
    cv_strategy="group_kfold",
    cv_groups=fs.metadata\["batch_id"\],  # Keep batches together
    n_splits=5
)

print(f"Batch-aware accuracy: {result.accuracy:.2%}")
print(f"Per-batch performance: {result.batch_metrics}")
```

**Output:** Realistic performance estimates respecting batch structure; per-batch accuracy breakdown showing generalization.

**Learn more:** [Validation Strategies](methods/validation/advanced_validation_strategies.md) • [Data Governance](user-guide/data_governance.md)

---

### 10. Detect Data Quality Issues

**Problem:** Dataset has missing values, outliers, or class imbalance that could bias results.

```python
from foodspec import FoodSpec

fs = FoodSpec("suspect_data.csv")

# Run automated quality checks
qa_report = fs.run_quality_checks(
    label_column="class",
    batch_column="batch_id",
    replicate_column="sample_id"
)

# View warnings
print(qa_report.warnings)
print(f"Class balance: {qa_report.class_balance}")
print(f"Leakage risk: {qa_report.leakage_score}")
```

**Output:** Quality report flagging: class imbalance, batch confounding, replicate leakage, missing data, outliers.

**Learn more:** [Data Governance Guide](user-guide/data_governance.md) • [MOATS Overview](theory/moats_overview.md)

---

### 11. Compare Preprocessing Methods

**Problem:** Unsure which preprocessing combination works best for your data.

```python
from foodspec import FoodSpec
from foodspec.validation import compare_preprocessing

fs = FoodSpec("raw_data.csv")

# Test multiple preprocessing pipelines
results = compare_preprocessing(
    fs,
    label_column="class",
    pipelines={
        "raw": [],
        "baseline_only": \[\{"method": "als", "lam": 1e6\}\],
        "baseline+norm": \[
            \{"method": "als", "lam": 1e6\},
            \{"method": "normalize", "norm": "snv"\}
        \],
        "full": \[
            \{"method": "als"\},
            \{"method": "smooth", "window": 11\},
            \{"method": "normalize", "norm": "snv"\}
        \]
    },
    cv_folds=5
)

print(results.comparison_table)
results.plot_comparison()
```

**Output:** Table comparing CV accuracy across pipelines; boxplot showing performance distribution; best pipeline recommendation.

**Learn more:** [Preprocessing Guide](methods/preprocessing/normalization_smoothing.md) • [Validation Cookbook](methods/validation/cross_validation_and_leakage.md)

---

### 12. Export Results for Publication

**Problem:** Need publication-ready figures and tables with proper metadata for methods section.

```python
from foodspec import FoodSpec

# Run analysis
fs = FoodSpec("final_data.csv")
result = fs.classify(label_column="class", model="pls-da")

# Export complete bundle
result.export_bundle(
    path="publication_outputs/",
    formats=\["png", "svg", "csv", "json"\],
    dpi=300,
    include_metadata=True,
    generate_report=True
)

# Auto-generate methods narrative
narrative = result.generate_methods_narrative(
    citation_style="APA",
    include_parameters=True
)
print(narrative)
```

**Output:** High-resolution figures (PNG/SVG), CSV tables, metadata.json (full reproducibility record), auto-generated methods text.

**Learn more:** [Reproducible Pipelines](tutorials/advanced/01-reproducible-pipelines.md) • [Workflow Design & Reporting](workflows/workflow_design_and_reporting.md)

---

## 📚 Next Steps

- **New to FoodSpec?** Start with [15-Minute Quickstart](getting-started/quickstart_15min.md)
- **Ready for full tutorials?** See [Tutorials Index](tutorials/index.md)
- **Need specific recipes?** Browse [Methods & Validation](methods/validation/index.md)
- **Building production systems?** Read [Theory & Advanced Methods](theory/index.md)

---

## 🔗 Quick Links by Domain

| Domain | Example Recipes | Full Workflows |
|--------|----------------|----------------|
| **Oil Authentication** | Recipe #1, #8 | [Oil Workflow](workflows/authentication/oil_authentication.md) |
| **Quality Monitoring** | Recipe #3, #10 | [Heating Quality](workflows/quality-monitoring/heating_quality_monitoring.md) |
| **Preprocessing** | Recipe #2, #4, #7, #11 | [Preprocessing Guide](methods/preprocessing/baseline_correction.md) |
| **Validation** | Recipe #9, #10 | [Validation Strategies](methods/validation/advanced_validation_strategies.md) |
| **Production** | Recipe #6, #12 | [Reproducible Pipelines](tutorials/advanced/01-reproducible-pipelines.md) |
| **Matrix Effects** | Recipe #5 | [Harmonization Theory](theory/harmonization_theory.md) |

---

**Keywords:** examples, recipes, quick start, code snippets, oil authentication, preprocessing, validation, reproducible pipelines, quality control
