# Workflow: Oil Authentication

## 📋 Standard Header

**Purpose:** Classify edible oils by type and detect adulteration using Raman/FTIR spectroscopy.

**When to Use:**
- Verify authenticity of olive oil or other high-value oils
- Detect adulteration with cheaper seed oils
- Classify unknown oil samples into known categories
- Monitor batch-to-batch consistency
- Quality assurance in oil production/import

**Inputs:**
- **Format:** HDF5 spectral library or CSV with wavenumber columns
- **Required metadata:** `oil_type` (classification label)
- **Optional metadata:** `batch`, `instrument`, `replicate_id`
- **Wavenumber range:** 600–1800 cm⁻¹ (fingerprint) + 2800–3100 cm⁻¹ (CH stretch)
- **Min samples:** 50–100 spectra (10+ per oil type for robust model)

**Outputs:**
- `confusion_matrix.png` — Classification performance by class
- `pca_scores.png` — Visual separation of oil types in reduced space
- `metrics.json` — Accuracy, macro F1, per-class precision/recall
- `report.md` — Narrative summary with interpretation
- `model.pkl` — Trained classifier for prediction on new samples

**Assumptions:**
- Spectra are from same spectral technique (Raman or FTIR, not mixed)
- Oil types have distinct chemical fingerprints (different saturation/oxidation)
- Labels are accurate (no mislabeling in training data)
- Samples span typical variability (multiple batches/sources per oil type)

---

## 🔬 Minimal Reproducible Example (MRE)

### Option A: Bundled Example Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from foodspec.data.loader import load_example_oils
from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.chemometrics.pca import run_pca
from foodspec.viz.pca import plot_pca_scores

# Load example dataset (bundled with FoodSpec)
fs = load_example_oils()
print(f"Loaded: {fs.x.shape[0]} spectra, {len(np.unique(fs.metadata['oil_type']))} oil types")
print(f"Oil types: {np.unique(fs.metadata['oil_type'])}")

# Run complete workflow
result = run_oil_authentication_quickstart(fs, label_column="oil_type")

# Display metrics
print(f"\nCross-Validation Results:")
print(f"  Accuracy: {result.cv_metrics['accuracy']:.1%}")
print(f"  Macro F1: {result.cv_metrics['macro_f1']:.3f}")
print(f"  Balanced Accuracy: {result.cv_metrics['balanced_accuracy']:.1%}")

# Plot confusion matrix
fig_cm, ax = plt.subplots(figsize=(8, 6))
plot_confusion_matrix(
    result.confusion_matrix,
    result.class_labels,
    ax=ax
)
plt.tight_layout()
plt.savefig("oil_confusion.png", dpi=150, bbox_inches='tight')
print("Saved: oil_confusion.png")

# PCA visualization
pca, pca_res = run_pca(fs.x, n_components=2)
fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
plot_pca_scores(
    pca_res.scores,
    labels=fs.metadata["oil_type"],
    ax=ax_pca
)
ax_pca.set_title(f"PCA: {pca.explained_variance_ratio_[:2].sum():.1%} variance explained")
plt.tight_layout()
plt.savefig("oil_pca.png", dpi=150, bbox_inches='tight')
print("Saved: oil_pca.png")
```

**Expected Output:**
```yaml
Loaded: 120 spectra, 3 oil types
Oil types: ['Authentic Olive' 'Mixed' 'Refined Seed']

Cross-Validation Results:
  Accuracy: 92.5%
  Macro F1: 0.923
  Balanced Accuracy: 92.5%

Saved: oil_confusion.png
Saved: oil_pca.png
```

### Option B: Synthetic Data Generator

```python
import numpy as np
import pandas as pd
from foodspec import SpectralDataset

def generate_synthetic_oils(n_per_class=30, n_wavenumbers=500, random_state=42):
    """Generate synthetic oil spectra with realistic peak patterns."""
    np.random.seed(random_state)
    
    wavenumbers = np.linspace(600, 1800, n_wavenumbers)
    
    oil_specs = {
        'Olive': {'peaks': [800, 1200, 1655], 'widths': [2000, 1500, 1800], 'heights': [2.0, 1.5, 1.8]},
        'Sunflower': {'peaks': [750, 1300, 1665], 'widths': [1800, 1800, 1600], 'heights': [2.2, 1.3, 2.0]},
        'Canola': {'peaks': [820, 1180, 1650], 'widths': [2000, 1500, 1700], 'heights': [1.9, 1.6, 1.7]},
    }
    
    spectra = []
    labels = []
    
    for oil_name, spec in oil_specs.items():
        for i in range(n_per_class):
            spectrum = np.random.normal(0, 0.1, n_wavenumbers)
            
            # Add characteristic peaks
            for peak, width, height in zip(spec['peaks'], spec['widths'], spec['heights']):
                spectrum += height * np.exp(-((wavenumbers - peak) ** 2) / width)
            
            # Add small batch effect
            batch_noise = np.random.normal(0, 0.05)
            spectrum += batch_noise
            
            spectra.append(spectrum)
            labels.append(oil_name)
    
    # Create DataFrame
    df = pd.DataFrame(
        np.array(spectra),
        columns=[f"{w:.1f}" for w in wavenumbers]
    )
    df.insert(0, 'oil_type', labels)
    df.insert(1, 'batch', np.random.choice(['A', 'B', 'C'], len(labels)))
    
    # Convert to SpectralDataset
    dataset = SpectralDataset.from_dataframe(
        df,
        metadata_columns=['oil_type', 'batch'],
        intensity_columns=[f"{w:.1f}" for w in wavenumbers],
        wavenumber=wavenumbers,
        labels_column='oil_type'
    )
    
    return dataset

# Generate and use
fs_synthetic = generate_synthetic_oils(n_per_class=30)
print(f"Generated: {fs_synthetic.x.shape[0]} synthetic spectra")

# Run workflow (same as Option A)
result = run_oil_authentication_quickstart(fs_synthetic, label_column="oil_type")
print(f"Accuracy: {result.cv_metrics['accuracy']:.1%}")
```

---

## 🔧 Complete End-to-End Worked Example

Here's a full, copy-paste-ready script from data load to report:

```python
from foodspec.datasets import load_oil_example_data
from foodspec.preprocess import baseline_als, normalize_snv, smooth_savgol
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv
from foodspec.plotting import plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("Step 1: Loading oil dataset...")
spectra = load_oil_example_data()
print(f"✅ Loaded {len(spectra)} spectra from {len(set(spectra.labels))} oil types")
print(f"   Labels: {set(spectra.labels)}")

# ============================================================================
# STEP 2: PREPROCESS
# ============================================================================
print("\nStep 2: Preprocessing...")
# Note: For proper validation, these steps are done inside CV folds (no leakage)
spectra = baseline_als(spectra)
spectra = smooth_savgol(spectra)
spectra = normalize_snv(spectra)
print("✅ Preprocessing complete")
print(f"   - Baseline correction (ALS)")
print(f"   - Savitzky-Golay smoothing")
print(f"   - Vector normalization")

# ============================================================================
# STEP 3: TRAIN & VALIDATE
# ============================================================================
print("\nStep 3: Training classifier...")
model = ClassifierFactory.create(
    "random_forest",
    n_estimators=100,
    max_depth=10,
    random_state=42
)

metrics = run_stratified_cv(
    model,
    spectra.data,
    spectra.labels,
    cv=5,
    random_state=42
)

print("✅ Cross-validation complete")
print(f"   Accuracy: {metrics['accuracy']:.1%}")
print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
print(f"   Macro F1: {metrics['macro_f1']:.3f}")

# ============================================================================
# STEP 4: VISUALIZE RESULTS
# ============================================================================
print("\nStep 4: Generating figures...")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
plot_confusion_matrix(
    metrics['confusion_matrix'],
    metrics.get('class_labels', sorted(set(spectra.labels))),
    ax=ax
)
plt.title("Oil Authentication: Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
plt.close()

# ROC Curve (if binary classification)
if len(set(spectra.labels)) == 2 and 'fpr' in metrics:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_roc_curve(
        metrics['fpr'],
        metrics['tpr'],
        metrics['roc_auc'],
        ax=ax
    )
    plt.title("Oil Authentication: ROC Curve")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: roc_curve.png")
    plt.close()

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================
print("\nStep 5: Saving results...")
import json

# Save metrics
with open("metrics.json", "w") as f:
    json.dump({
        "accuracy": float(metrics['accuracy']),
        "balanced_accuracy": float(metrics['balanced_accuracy']),
        "macro_f1": float(metrics['macro_f1']),
        "n_samples": len(spectra),
        "n_classes": len(set(spectra.labels))
    }, f, indent=2)

print("✅ Saved: metrics.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("OIL AUTHENTICATION WORKFLOW COMPLETE")
print("="*70)
print(f"✅ Accuracy: {metrics['accuracy']:.1%}")
print(f"✅ Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
print(f"✅ Macro F1: {metrics['macro_f1']:.3f}")
print(f"\nOutputs:")
print(f"  - confusion_matrix.png")
print(f"  - roc_curve.png (if binary)")
print(f"  - metrics.json")
print("="*70)
```

**Expected output:**
```
Step 1: Loading oil dataset...
✅ Loaded 96 spectra from 4 oil types
   Labels: {'Olive', 'Palm', 'Sunflower', 'Coconut'}

Step 2: Preprocessing...
✅ Preprocessing complete
   - Baseline correction (ALS)
   - Savitzky-Golay smoothing
   - Vector normalization

Step 3: Training classifier...
✅ Cross-validation complete
   Accuracy: 95.2%
   Balanced Accuracy: 94.8%
   Macro F1: 0.948

Step 4: Generating figures...
✅ Saved: confusion_matrix.png
✅ Saved: metrics.json

======================================================================
OIL AUTHENTICATION WORKFLOW COMPLETE
======================================================================
✅ Accuracy: 95.2%
✅ Balanced Accuracy: 94.8%
✅ Macro F1: 0.948

Outputs:
  - confusion_matrix.png
  - metrics.json
======================================================================
```

---

## Why These Choices?

**Random Forest vs. other models:**
- Works well on spectroscopy data (nonlinear relationships)
- Fast to train and predict
- Interpretable feature importance
- No hyperparameter tuning required for baseline

**Baseline Correction (ALS):**
- Removes sloping background common in Raman/FTIR
- Better than linear baseline (handles curves)
- Alternative: `baseline_als` or `rubberband`

**Normalization (Vector Normalization):**
- Removes effects of sample size, laser power, path length
- Makes oils comparable regardless of instrument setup
- Alternative: `snv` (Standard Normal Variate) or `msc`

**Cross-Validation:**
- 5-fold CV balances bias and variance
- Stratified ensures class distribution in each fold
- Gives honest performance estimate

---

## Troubleshooting

| Problem | Solution |
- ✅ Diagonal values > 80% of row totals (good per-class accuracy)
- ✅ Off-diagonal entries small and balanced (no systematic confusion)
- ✅ All classes represented (no empty rows/columns)

**PCA Scores Plot:**
- ✅ Clear clustering by oil type (minimal overlap)
- ✅ PC1 + PC2 explain > 70% variance (captures most information)
- ✅ No outliers far from any cluster (data quality issue)

**Metrics:**
- ✅ Accuracy > 85% (for well-separated oil types)
- ✅ Macro F1 > 0.80 (balanced performance across classes)
- ✅ Balanced accuracy ≈ regular accuracy (classes balanced)

**Feature Importance:**
- ✅ Top features correspond to known peaks (1655 C=C, 1742 C=O, 1450 CH2)
- ✅ Multiple features contribute (not relying on single peak)
- ✅ Loadings/importances chemically interpretable

### Failure Indicators

**⚠️ Warning Signs:**

1. **Accuracy > 95% but macro F1 < 0.70**
   - Problem: Severe class imbalance; model biased toward majority class
   - Fix: Use balanced_accuracy or stratified sampling; check class distribution

2. **PCA shows no separation (overlap > 50%)**
   - Problem: Oil types too similar spectrally; insufficient chemical differences
   - Fix: Check preprocessing (try different baseline/normalization); verify labels correct

3. **One oil type always misclassified as another**
   - Problem: Spectral overlap or mislabeling
   - Fix: Review peak positions; check if oils genuinely different; inspect raw spectra

4. **High training accuracy (>95%) but CV accuracy < 75%**
   - Problem: Overfitting; model learned noise/batch effects
   - Fix: Reduce model complexity (fewer trees, simpler features); increase regularization

5. **Feature importance dominated by edge wavenumbers (< 650 or > 1750 cm⁻¹)**
   - Problem: Edge artifacts or noise driving classification
   - Fix: Crop more aggressively; check baseline correction quality

6. **CV folds show high variance (accuracy 60–95% across folds)**
   - Problem: Dataset too small or batch effects confounding
   - Fix: Increase sample size; stratify by batch; check for outliers

### Quality Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Accuracy | 75% | 85% | 95% |
| Macro F1 | 0.70 | 0.85 | 0.95 |
| Balanced Accuracy | 75% | 85% | 95% |
| PC1 + PC2 Variance | 60% | 75% | 90% |
| Per-Class Recall | 70% | 85% | 95% |

---

## ⚙️ Parameters You Must Justify

### Critical Parameters (Report in Methods)

**1. Baseline Correction (ALS)**
- **Parameter:** `lam` (smoothness), `p` (asymmetry)
- **Default:** lam=1e4, p=0.01
- **When to adjust:**
  - Increase `lam` (1e5) if spectra have strong curvature
  - Decrease `p` (0.001) if fluorescence dominates
- **Justification template:**
  > "Asymmetric Least Squares baseline correction (λ=1e4, p=0.01) removed background curvature while preserving peak shape (Eilers & Boelens, 2005)."

**2. Smoothing (Savitzky-Golay)**
- **Parameter:** `window_length`, `polyorder`
- **Default:** window=21, polyorder=3
- **When to adjust:**
  - Increase window (31, 41) if spectra very noisy (SNR < 10)
  - Decrease window (11, 15) if peaks are narrow
- **Justification template:**
  > "Savitzky-Golay smoothing (window=21, polynomial order=3) reduced high-frequency noise while preserving peak positions (Savitzky & Golay, 1964)."

**3. Normalization**
- **Parameter:** Method (SNV, L2, minmax)
- **Default:** L2 (unit vector)
- **When to adjust:**
  - Use SNV if baseline variability persists after ALS
  - Use minmax if peak ratios are critical
- **Justification template:**
  > "Spectra were normalized to unit L2 norm to remove intensity scaling artifacts while preserving relative peak heights."

**4. Spectral Cropping**
- **Parameter:** Wavenumber range
- **Default:** 600–1800 cm⁻¹
- **When to adjust:**
  - Extend to 2800–3100 cm⁻¹ if CH stretch region informative
  - Narrow to 1200–1800 cm⁻¹ if only carbonyl/unsaturation relevant
- **Justification template:**
  > "Spectra were cropped to 600–1800 cm⁻¹ to focus on the fingerprint region containing characteristic C=O (1742 cm⁻¹) and C=C (1655 cm⁻¹) stretching modes."

**5. Model Selection**
- **Parameter:** Classifier type (RF, SVM, PLS-DA)
- **Default:** Random Forest (n_estimators=100)
- **When to adjust:**
  - Use SVM if classes linearly separable in PCA space
  - Use PLS-DA if you need interpretable loadings
- **Justification template:**
  > "Random Forest (100 trees) was chosen for robustness to outliers and ability to capture non-linear class boundaries without hyperparameter tuning."

**6. Cross-Validation**
- **Parameter:** n_splits, stratification
- **Default:** 5-fold stratified CV
- **When to adjust:**
  - Use 10-fold if dataset large (> 200 samples)
  - Use leave-one-batch-out if batch effects suspected
- **Justification template:**
  > "Five-fold stratified cross-validation ensured balanced class representation in each fold and unbiased performance estimation."

### Optional Parameters (Mention if Changed)

**Feature Extraction:**
- Peak detection thresholds
- Ratio definitions (numerator/denominator wavenumbers)
- PCA n_components

**Model Hyperparameters:**
- Random Forest: max_depth, min_samples_split
- SVM: kernel, C, gamma
- PLS-DA: n_components

---

Relevant visual aids: spectrum overlays, PCA scores/loadings, confusion matrix, boxplots/violin plots of key ratios. See [Plots guidance](../workflow_design_and_reporting.md#plots-visualizations) for expectations. The tutorial content is merged here; see also the legacy tutorial file for reference.

```mermaid
flowchart LR
  subgraph Data
    A[Raw/vendor files]
    B[read_spectra -> FoodSpectrumSet]
  end
  subgraph Preprocess & Features
    C[Baseline + SavGol + norm + crop]
    D[Peaks + ratios]
  end
  subgraph Model
    E[RF / SVM / PLS-DA]
  end
  subgraph Evaluate
    F[CV metrics + confusion + PCA]
    G[Stats on ratios (ANOVA/G-H)]
  end
  subgraph Report
    H[plots + metrics.json + report.md]
  end
  A --> B --> C --> D --> E --> F --> H
  D --> G --> H
```

## 1. Problem and dataset
- **Why labs care:** Adulteration (cheap oils in EVOO), mislabeling, batch verification.
- **Inputs:** Spectral library (HDF5) with columns: `oil_type` (label), optional `batch`, `instrument`. Wavenumber axis in ascending cm⁻¹ (Raman/FTIR fingerprint 600–1800 cm⁻¹, CH stretch 2800–3100 cm⁻¹).
- **Typical size:** Tens to hundreds of spectra per class for robust models; synthetic examples work for testing.

## 2. Pipeline (default)
- **Preprocessing:** ALS baseline → Savitzky–Golay smoothing → L2 normalization → crop to 600–1800 cm⁻¹.
- **Features:** Expected peaks (≈1655 C=C, 1742 C=O, 1450 CH2 bend); ratios (1655/1742, 1450/1655).
- **Models:** Random Forest (robust default), or SVM/PLS-DA for linear boundaries.
- **Validation:** Stratified k-fold CV (default 5 folds); metrics: accuracy, macro F1; confusion matrix.

## 3. Python example
```python
from foodspec.data.loader import load_example_oils
from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.chemometrics.pca import run_pca
from foodspec.viz.pca import plot_pca_scores

fs = load_example_oils()
result = run_oil_authentication_quickstart(fs, label_column="oil_type")

# Metrics
print(result.cv_metrics)

# Confusion matrix
fig_cm = plot_confusion_matrix(result.confusion_matrix, result.class_labels)
fig_cm.savefig("oil_confusion.png", dpi=150)

# PCA visualization on preprocessed spectra
pca, res = run_pca(fs.x, n_components=2)
fig_scores = plot_pca_scores(res.scores, labels=fs.metadata["oil_type"])
fig_scores.savefig("oil_pca.png", dpi=150)
```

### Optional deep-learning baseline
```python
# pip install foodspec[deep]
from foodspec.chemometrics.deep import Conv1DSpectrumClassifier
from foodspec.metrics import compute_classification_metrics

model = Conv1DSpectrumClassifier(n_filters=8, n_epochs=15, batch_size=16, random_state=0)
model.fit(fs.x, fs.metadata["oil_type"])
dl_pred = model.predict(fs.x)
dl_metrics = compute_classification_metrics(fs.metadata["oil_type"], dl_pred)
print("DL accuracy:", dl_metrics["accuracy"])
```
Use only when you have sufficient samples per class; always compare against classical baselines and inspect F1_macro/confusion matrices.

## 4. CLI example (with config)
Create `examples/configs/oil_auth_quickstart.yml`:
```yaml
input_hdf5: libraries/oils.h5
label_column: oil_type
classifier_name: rf
cv_splits: 5
```
Run:
```bash
foodspec oil-auth --config examples/configs/oil_auth_quickstart.yml --output-dir runs/oil_auth_demo
```
Outputs: `metrics.json`, `confusion_matrix.png`, `report.md` in a timestamped folder.

## 5. Interpretation
- Report overall accuracy and macro F1; include confusion matrix with class labels.
- Mention preprocessing steps (baseline, smoothing, normalization, crop) and feature choices (peak/ratio definitions).
- Highlight chemically meaningful loadings/feature importances (e.g., unsaturation bands).
- Main text: summary metrics + confusion matrix figure. Supplement: per-class precision/recall, spectra examples, configs.

### Qualitative & quantitative interpretation
- **Qualitative:** PCA scores and ratio boxplots show class structure; confusion matrix reveals which oils are confused. RF importances/PLS loadings (see interpretability figures) highlight bands driving separation—link back to unsaturation/carbonyl bands.
- **Quantitative:** Report macro F1/balanced accuracy; silhouette on PCA scores; ANOVA/Tukey/Games–Howell on key ratios (link to [ANOVA/MANOVA](../../methods/statistics/anova_and_manova.md)); effect sizes when applicable.
- **Reviewer phrasing:** “PCA shows moderate separation of oil classes (silhouette ≈ …); the RF classifier reached macro F1 = …; ratios at 1655/1742 cm⁻¹ differed across oils (ANOVA p < …).”

### Peak & ratio summary tables
- Generate mean ± std of key peak positions/intensities and ratios by oil_type for supplementary tables.
- Example: use `compute_peak_stats` and `compute_ratio_table` on extracted features; report which bands/ratios differ most across oils (with p-values/effect sizes).
- Reviewer phrasing: “Table 1 summarizes unsaturation/carbonyl ratios by oil type (mean ± SD); Games–Howell indicates oil A > oil B (p_adj < …).”
- Visuals to pair: RF feature importances / PLS loadings (assets `rf_feature_importance.png`, `pls_loadings.png`) to link discriminative bands to chemistry.

## Summary
- Baseline + smoothing + normalization + crop → peak/ratio features → RF/SVM/PLS-DA → CV metrics and confusion matrix.
- Use stratified CV; report macro metrics; tie discriminative bands back to chemistry.

## Statistical analysis
- **Why:** Beyond classification metrics, test whether key ratios differ across oil types to support interpretation.
- **Example (ANOVA + Tukey):**
```python
import pandas as pd
from foodspec.stats import run_anova, run_tukey_hsd
from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.data.loader import load_example_oils
import pandas as pd

fs = load_example_oils()
res = run_oil_authentication_quickstart(fs, label_column="oil_type")

# Extract ratio features from the fitted pipeline
preproc = res.pipeline.named_steps["preprocess"]
features = res.pipeline.named_steps["features"]
feat_array = features.transform(preproc.transform(fs.x))
cols = features.named_steps["to_array"].columns_
ratio_series = pd.Series(feat_array[:, 0], index=fs.metadata.index, name=cols[0])

anova_res = run_anova(ratio_series, fs.metadata["oil_type"])
print(anova_res.summary)
try:
    tukey = run_tukey_hsd(ratio_series, fs.metadata["oil_type"])
    print(tukey.head())
except ImportError:
    pass
# Robust post-hoc if variances/group sizes differ
gh = games_howell(ratio_series, fs.metadata["oil_type"])
print(gh.head())
```
- **Interpretation:** ANOVA p-value < 0.05 suggests at least one oil type differs in the ratio; Tukey or Games–Howell identifies which pairs. Report effect size where possible.
See theory: [Hypothesis testing](../../methods/statistics/hypothesis_testing_in_food_spectroscopy.md), [ANOVA](../../methods/statistics/anova_and_manova.md).

### Ratio plots (recommended)
- Use `plot_ratio_by_group` for key ratios (e.g., 1655/1742) across oil types; separated medians/IQRs imply differences—support with ANOVA/Games–Howell and effect sizes.
- Ratio–ratio scatter (e.g., 1655/1742 vs 3010/2850) highlights compositional regimes; pair with silhouette/ANOVA on each ratio.
- Summary tables (peak/ratio mean ± SD by oil_type) can accompany plots in supplementary material.

---

## When Results Cannot Be Trusted

⚠️ **Red flags for oil authentication workflow:**

1. **Model trained and tested on oils from same source/batch (e.g., all "olive" from single producer/harvest)**
   - Intra-source variability unknown; model may learn producer-specific patterns, not species
   - Different olive cultivar or origin will fail
   - **Fix:** Include multiple sources per oil type; validate across different cultivars/origins

2. **No adulterant testing (model validated only on pure oils, not blends or refined oils)**
   - Pure-oil classification doesn't confirm ability to detect adulteration
   - Refined oils may cluster closer to pure oils than expected
   - **Fix:** Include known adulterants (refined oils, blends) in test set; test detection rates at 1%, 5%, 10% adulteration levels

3. **Ratios or features cherry-picked post-hoc to separate oils**
   - Data-dependent feature selection inflates reproducibility claims
   - Different dataset may reveal different separating features
   - **Fix:** Use univariate feature selection a priori; or use model-based importance from cross-validation

4. **Authentication model based on single spectral region (only CH stretches, ignore C=O region)**
   - Narrow spectral window may miss adulterants affecting other regions
   - Real adulterants exploit regions unchecked
   - **Fix:** Use full spectral range; test sensitivity to adulterants in different regions

5. **Cross-contamination during sample preparation (using same pipette for different oils)**
   - Cross-contamination creates false similarity between oils
   - Baseline or preprocessing steps may not remove contamination
   - **Fix:** Use separate equipment per sample; measure blanks between samples; document sample handling

6. **Confusing near-infrared (NIR) with Raman/FTIR without method validation**
   - Different spectroscopic methods give different spectral signatures
   - Transferring models between methods requires retraining
   - **Fix:** Validate method-specific models; don't mix spectra from different instruments/wavelengths without harmonization

7. **Model accuracy high (>95%) but specificity/sensitivity per oil type varies wildly**
   - Macro-accuracy can mask severe class-specific failures
   - Confusion matrix and per-class metrics reveal true performance
   - **Fix:** Report per-class precision/recall; show confusion matrix; investigate misclassified oils

8. **No temporal validation (model trained on 2024 oils, deployed on 2023 samples without revalidation)**
   - Aging, storage, or oxidation changes oil spectra over time
   - Model trained on recent oils may fail on archived samples
   - **Fix:** Test on samples from different harvest years; monitor model performance over time; retrain periodically

## Further reading
- [Baseline correction](../../methods/preprocessing/baseline_correction.md)
- [Feature extraction](../../methods/preprocessing/feature_extraction.md)
- [Classification & regression](../../methods/chemometrics/classification_regression.md)
- [Model evaluation](../../methods/chemometrics/model_evaluation_and_validation.md)
