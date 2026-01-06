# Why FoodSpec? Statement of Need and Design Philosophy

---

## Statement of Need

Food scientists, chemometricians, and QC labs rely on vibrational spectroscopy (Raman, FTIR, NIR) to detect adulteration, track degradation, and verify authenticity. Today, these workflows are fragmented:

1. **Import** via vendor software (Bruker OPUS, Thermo Galactic, Agilent)
2. **Preprocess** in ad hoc scripts or separate tools (baseline, normalization, smoothing)
3. **Extract features** in notebooks or Excel
4. **Model** in R/Python/MATLAB with custom train-test splits
5. **Validate** often without leakage checks or nested CV
6. **Report** manually, without provenance or reproducibility metadata

The result: analyses are hard to repeat, impossible to audit, and vulnerable to silent leakage (e.g., preprocessing before splitting, batch effects ignored). FoodSpec consolidates these steps into a single, defensible pipeline with reproducibility built in.

---

## Problems in Food Spectroscopy FoodSpec Addresses

### 1. No Standard End-to-End Workflow

**Problem**: Each lab builds preprocessing pipelines from scratch. Oil authentication in one lab uses ALS baseline + SNV, another uses rubberband + MSC—with no shared defaults or validation patterns.

**FoodSpec solution**: Peer-reviewed workflows (oil authentication, heating degradation, QC) encode best practices. Users can extend or override defaults, but sensible food-specific choices come out of the box.

```python
# Instead of writing baseline → norm → split → train → validate logic:
from foodspec.workflows.oils import run_oil_authentication_workflow

result = run_oil_authentication_workflow(
    dataset,
    label_column="oil_type",
    preprocessing=["baseline_als", "normalize_snv"],
    cv_folds=10
)
```

### 2. Data Leakage and Poor Validation Practices

**Problem**: Many published food spectroscopy studies:
- Preprocess before splitting (leakage)
- Split replicates across train/test without grouping
- Use single 70/30 split instead of nested CV
- Report accuracy without uncertainty or calibration diagnostics

**FoodSpec solution**: Preprocessing happens inside CV folds by default. Batch/replicate groups are kept together. Nested CV gives unbiased performance estimates. Uncertainty (Brier score, calibration plots) is computed automatically.

```python
# Preprocessing locked inside CV folds; no leakage risk
result = fs.validate_with_nested_cv(
    preprocessing=["baseline_als", "normalize_snv"],
    n_outer_folds=10,
    n_inner_folds=5,
    stratify_by="oil_type",
    group_by="batch"  # keeps batch together across folds
)
print(result.balanced_accuracy, result.ci_95)  # confidence interval included
```

### 3. Vendor Lock-In and Format Chaos

**Problem**: OPUS files, SPA files, DPT files, CSV exports—each has quirks (header rows, column order, metadata loss). Moving data between labs or instruments requires error-prone manual conversion.

**FoodSpec solution**: Unified import layer handles OPUS, SPA, DPT, CSV, HDF5. Metadata (instrument, operator, date) is preserved. Export to HDF5 for efficient storage and reproducible access.

```bash
# One-liner import from OPUS, SPA, DPT, or CSV
foodspec csv-to-library raw_spectra.csv library.h5 \
  --wavenumber-col wavenumber --sample-id-col sample_id
```

### 4. No Domain-Specific Interpretability

**Problem**: Generic PCA/PLS tools do not know that oils need band ratio checking (1650/1440 cm⁻¹), or that heating studies need time-series validation. Features and thresholds are generic.

**FoodSpec solution**: Food-specific feature extractors include peak detection with chemical band libraries, ratiometric analysis with domain defaults, and time-aware validation for stability studies.

```python
# Extract oil-relevant band ratios automatically
peaks = extract_peaks(spectrum, prominence=0.05)
ratio_1650_1440 = peaks["1650_cm"] / peaks["1440_cm"]
# or use food defaults:
feature_set = extract_oil_authentication_features(spectrum)
```

### 5. Missing Reproducibility Infrastructure

**Problem**: Analyses are not reproducible because:
- Parameter choices are scattered in notebooks
- Software versions are not logged
- Figures are regenerated without metadata
- No audit trail for what changed between runs

**FoodSpec solution**: YAML protocols capture all choices (preprocessing method, hyperparameters, CV strategy, figures). Every run generates artifacts (figures, metrics, metadata, methods text). Provenance logs include versions, timestamps, and data checksums.

```yaml
# reproducible_protocol.yaml
protocol: oil_authentication
data: {library: oils_raman.h5, label_column: oil_type}
preprocessing:
  - {method: baseline_als, lambda: 1e5}
  - {method: normalize_snv}
validation: {cv_strategy: stratified_kfold, n_folds: 10, nested: true}
model: {algorithm: random_forest, n_estimators: 100}
output: {figures: [confusion_matrix, roc_curve, feature_importance]}
```

---

## How FoodSpec Differs from Related Tools

FoodSpec is not a replacement; rather, it is a specialized layer built on top of general-purpose tools.

| Aspect | ChemoSpec (R) | HyperSpy (Python) | Spectral Python (SPy) | FoodSpec |
|--------|---------------|-------------------|-----------------------|----------|
| **Primary use** | Multivariate analysis (PCA, PLS) | Hyperspectral preprocessing & mapping | Hyperspectral classification | End-to-end food workflows |
| **Data model** | Matrix/dataframe | Lazy HyperSpectralSignal objects | Array-based | SpectralDataset, HyperSpectralCube, MultiModalDataset |
| **Baseline correction** | No | Basic (polynomial, rolling ball) | No | 6 methods (ALS, rubberband, polynomial, airPLS, morphological, rolling ball) |
| **Food-specific workflows** | No | No | No | Yes (oil auth, heating, QC, mixture analysis) |
| **Validation** | Basic (train/test split) | Not focused | Not focused | Nested CV, leakage detection, batch grouping |
| **Reproducibility** | Manual logging | Manual logging | Manual logging | YAML protocols, full provenance, auto-reporting |
| **CLI & automation** | No | No | No | Yes (batch processing, scheduled runs) |

### What Each Tool Does Well

- **ChemoSpec**: Excellent for exploratory multivariate analysis; strong statistics. Use it if you want interactive PCA/PLS in R.
- **HyperSpy**: Best-in-class for hyperspectral preprocessing and visualization. Use it for detailed HSI preprocessing before feeding data to FoodSpec.
- **Spectral Python (SPy)**: Solid for hyperspectral classification and target detection. Use it for pixel-level supervised learning on HSI.
- **FoodSpec**: Streamlined end-to-end workflows for food authenticity, quality, and calibration. Use it when you need leakage-free validation and domain-specific defaults.

### Interoperability

FoodSpec integrates with these tools:

```python
# Read HyperSpy data into FoodSpec
hsi = hs.load("hyperspec.hspy")
dataset = FoodSpec.from_hyperspy(hsi)

# Export to ChemoSpec-compatible CSV for PCA comparison
dataset.to_csv("for_chemospec.csv")

# Feed FoodSpec-preprocessed data to scikit-learn
X_clean = dataset.preprocess(methods=["baseline_als", "normalize_snv"])
clf = RandomForestClassifier().fit(X_clean, dataset.labels)
```

---

## Reproducibility Philosophy

FoodSpec's reproducibility strategy rests on three pillars:

### 1. **Protocol as Code**

All choices (preprocessing, validation, modeling) are captured in YAML and versioned with your code. No clicking buttons in GUIs; no hidden parameters.

```yaml
protocol: oil_authentication
preprocessing:
  - method: baseline_als
    lambda: 1e5
    max_iters: 100
```

### 2. **Full Provenance**

Every run logs:
- Software versions (FoodSpec, dependencies)
- Data checksums (verify inputs unchanged)
- Parameter snapshots (no guessing what was set)
- Timestamp and operator name
- Metrics and figures

```
Run ID: 20260106_oil_auth_batch3
FoodSpec version: 1.0.0
Data checksum: sha256:a3f5c9e...
Baseline method: ALS (lambda=1e5)
Balanced accuracy: 0.94 (CI: 0.88–0.98)
Created: 2026-01-06 14:32 UTC by alice@lab.org
Artifacts: results/confusion_matrix.png, results/metrics.json
```

### 3. **Immutable Artifacts**

All outputs (models, figures, metrics) are generated fresh from the protocol. No manual editing of figures; no ad hoc tweaks. If you change a parameter, you re-run the full pipeline.

```python
# Run once; results are immutable
fs.run_protocol("oil_auth.yaml", output_dir="results/")
# results/
#   ├── confusion_matrix.png
#   ├── roc_curve.png
#   ├── metrics.json
#   ├── model.pkl
#   └── metadata.json (full provenance)
```

---

## Decision Tree: Which FoodSpec Feature Should I Use?

```
Your goal: Authenticate oils for adulteration detection
├─ Start with: oil_authentication workflow (tutorials/intermediate/01-oil-authentication.md)
├─ Preprocessing: Use ALS baseline + SNV (food defaults)
└─ Validation: Nested CV, stratified splits, batch-aware

Your goal: Track oxidation in heated oils over time
├─ Start with: heating_degradation workflow
├─ Preprocessing: Baseline ALS, normalize SNV
├─ Validation: Time-aware splits (older samples as train, newer as test)
└─ Features: 1650/1440 band ratio, CH saturation index

Your goal: Mixture analysis (binary blend, e.g., olive + sunflower)
├─ Start with: mixture_analysis workflow
├─ Method: MCR-ALS or NNLS
├─ Preprocessing: Baseline, normalization, optional smoothing
└─ Output: Component concentration estimates + uncertainty

Your goal: Hyperspectral mapping (spatial analysis)
├─ Start with: tutorials/advanced/03-hsi-mapping.md
├─ Load: HyperSpectralCube from datacube or directory
├─ Preprocessing: Baseline per pixel, normalization
├─ Model: PCA projection to visualize composition across image
└─ Export: False-color composites; labeled mask overlays

Your goal: Custom preprocessing (e.g., compare 3 baseline methods)
├─ Use: SpectralDataset API
├─ Load: dataset = SpectralDataset.from_csv("data.csv")
├─ Preprocess: X1 = dataset.apply_baseline("als"), X2 = dataset.apply_baseline("rubberband"), ...
├─ Compare: Plot side-by-side, compute residuals
└─ Validate: Apply to test set, check metrics

Your goal: Build a QC pipeline for batch screening
├─ Use: CLI (foodspec cli --help) or Python API
├─ Load: foodspec csv-to-library raw.csv library.h5
├─ Define: YAML protocol with thresholds
├─ Automate: foodspec run-protocol qc_batch.yaml --input library.h5
└─ Report: Auto-generate pass/fail summary

Your goal: Deploy model to production / QC lab
├─ Start with: developer-guide/model_deployment.md
├─ Save: trained_model = fs.train_model(protocol)
├─ Serialize: trained_model.save("model_v1.pkl")
├─ Load: loaded_model = load_model("model_v1.pkl")
├─ Predict: predictions = loaded_model.predict(new_data)
└─ Log: Every prediction logged with timestamp, data ID, confidence

Your goal: Understand leakage and validation
├─ Read: methods/validation/cross_validation_and_leakage.md
├─ Key concept: Preprocessing must happen inside CV folds
├─ FoodSpec default: Enforces this automatically
└─ Check: Run fs.check_for_leakage(dataset) before modeling

Your goal: Compare FoodSpec to R ChemoSpec
├─ Export: dataset.to_csv("for_r.csv")
├─ Import in R: data <- read.csv("for_r.csv"); PCA <- prcomp(...)
├─ FoodSpec advantage: Validated preprocessing + domain workflows
└─ ChemoSpec advantage: Interactive visualization, advanced stats

Your goal: Extend FoodSpec (custom preprocessing, workflow)
├─ Start with: developer-guide/extending_protocols_and_steps.md
├─ Write: Custom PreprocessingStep subclass
├─ Register: Register via registry.register_preprocessing_step(...)
├─ Use: reference in YAML as custom:my_method
└─ Share: Open an issue or PR; we love contributions
```

---

## When to Use FoodSpec vs. When to Keep Your Existing Stack

### ✅ Use FoodSpec If:

- You analyze Raman, FTIR, or NIR data in food science
- You need reproducible, leakage-free validation
- You want to share analyses with colleagues (same code, same results)
- You need automation (batch processing, scheduled runs)
- You want built-in domain expertise (oil auth, heating, QC workflows)
- Regulatory or publication standards require provenance and methodology transparency

### ❌ Keep Your Existing Stack If:

- You use certified vendor software (ISO/FDA/AOAC methods); FoodSpec is decision support, not a replacement
- Your data is non-vibrational (e.g., chromatography, immunoassays)
- You need real-time edge computing; FoodSpec targets server/desktop environments
- You are doing pure exploratory analysis on a single dataset and do not need reproducibility
- Your team is heavily invested in a different ecosystem (R/MATLAB) and switching is not worth it

---

## Quick Example: Why It Matters

### Without FoodSpec (Typical Lab Workflow)

```python
# data.csv: wavenumber, oil_type, sample1, sample2, ...
df = pd.read_csv("data.csv")
X = df.iloc[:, 2:].T  # samples × wavenumber
y = df.iloc[:, 1]

# Whoops: preprocessed before splitting!
from scipy.ndimage import gaussian_filter1d
X_smooth = gaussian_filter1d(X, sigma=2, axis=1)
# Fit baseline...
# Normalize...

# Split
X_train, X_test, y_train, y_test = train_test_split(X_smooth, y, test_size=0.2, random_state=42)

# Train
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"Accuracy: {acc:.3f}")

# Reviewer asks: "Did you preprocess before or after splitting?"
# Answer: ...😬 (leakage!)
```

### With FoodSpec

```python
from foodspec import SpectralDataset
from foodspec.workflows.oils import run_oil_authentication_workflow

ds = SpectralDataset.from_csv("data.csv", label_column="oil_type")

result = run_oil_authentication_workflow(
    ds,
    preprocessing=["baseline_als", "normalize_snv"],
    cv_strategy="stratified_kfold",
    n_folds=10,
    nested=True  # nested CV for unbiased estimate
)

print(f"Balanced Accuracy: {result.balanced_accuracy:.3f}")
print(f"95% CI: {result.ci_95}")
print(f"Confusion matrix:\n{result.confusion_matrix}")

# Save protocol for reproducibility
result.save_protocol("oil_auth_protocol.yaml")
result.save_artifacts("results/")

# Reviewer asks: "Can you re-run this analysis?"
# Answer: "Sure! foodspec run-protocol oil_auth_protocol.yaml"
# Results are identical ✅
```

---

## Summary

FoodSpec exists because food spectroscopy workflows need:

1. **Integration**: Import → preprocess → feature → model → validate in one place
2. **Leakage-free defaults**: CV folds contain preprocessing; batch/replicate grouping enforced
3. **Domain expertise**: Oil auth, heating, QC workflows with food-specific choices
4. **Reproducibility**: YAML protocols, provenance logs, immutable artifacts
5. **Simplicity**: Users write 5 lines of code instead of 500

It complements (not replaces) general-purpose tools like ChemoSpec, HyperSpy, and Spectral Python.

---

## Next Steps

- **New to FoodSpec?** → [15-Minute Quickstart](../getting-started/quickstart_15min.md)
- **Want a full walkthrough?** → [Oil Authentication Tutorial](../tutorials/intermediate/01-oil-authentication.md)
- **Curious about methods?** → [Preprocessing Guide](../methods/preprocessing/baseline_correction.md)
- **Ready to deploy?** → [Developer Guide](../developer-guide/contributing.md)
- **Questions?** → [FAQ](../help/faq.md)
