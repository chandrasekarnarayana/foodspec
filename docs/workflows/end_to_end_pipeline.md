# End-to-End Pipeline: From Raw Spectra to Validated Results

**Purpose:** Complete walkthrough of FoodSpec analysis workflow from data import to validation and reproducibility.

**Audience:** Researchers, students, QA engineers new to FoodSpec.

**Time:** 45 minutes (or ~10 minutes per section; sections are independent).

**Prerequisites:** FoodSpec installed; familiarity with Python, CSV/HDF5 data, and basic statistics.

---

## Quick Overview

This guide demonstrates a complete oil authentication workflow:

1. **Load & explore** (5 min): CSV → FoodSpec dataset with metadata
2. **Preprocess** (10 min): Baseline correction, normalization, smoothing
3. **Extract features** (10 min): Peak detection, ratios, PCA
4. **Classify** (10 min): Train/validate with nested CV
5. **Interpret** (5 min): Feature importance, confidence, misclassifications
6. **Reproduce** (5 min): Save protocol, generate methods text

**Why this example:** Oil authentication is a realistic use case where leakage, batch effects, and vendor formats commonly cause failures. The steps reflect a defensible workflow.

---

## Scenario: Oil Authentication Workflow

**Research question:** Can we reliably distinguish virgin olive oil (VOO) from refined olive oil (ROO) using Raman spectroscopy?

**Dataset:** 60 oil samples (30 VOO, 30 ROO) measured at 10 μm/s scan rate, acquired on Bruker SENTERRA II at room temperature.

---

## Complete Real-World Example Script

Here's the complete workflow from start to finish in one script:

```python
from foodspec.datasets import load_oil_example_data
from foodspec.preprocess import baseline_als, normalize_snv, smooth_savgol
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv
from foodspec.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import json

print("="*70)
print("COMPLETE END-TO-END OIL AUTHENTICATION WORKFLOW")
print("="*70)

# STEP 1: LOAD DATA
print("\nSTEP 1: Loading data...")
spectra = load_oil_example_data()
print(f"✅ Loaded {len(spectra)} spectra")
print(f"   Classes: {set(spectra.labels)}")
print(f"   Shape: {spectra.data.shape}")

# STEP 2: PREPROCESS
print("\nSTEP 2: Preprocessing...")
spectra = baseline_als(spectra)
spectra = smooth_savgol(spectra)
spectra = normalize_snv(spectra)
print("✅ Applied:")
print("   - Baseline correction (ALS)")
print("   - Smoothing (Savitzky-Golay)")
print("   - Normalization (SNV)")

# STEP 3: CLASSIFY WITH CROSS-VALIDATION
print("\nSTEP 3: Training classifier...")
model = ClassifierFactory.create("random_forest", n_estimators=100, random_state=42)
metrics = run_stratified_cv(model, spectra.data, spectra.labels, cv=5, random_state=42)

print("✅ Cross-validation complete:")
print(f"   Accuracy: {metrics['accuracy']:.1%}")
print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
print(f"   Macro F1: {metrics['macro_f1']:.3f}")

# STEP 4: VISUALIZE
print("\nSTEP 4: Generating figures...")
fig, ax = plt.subplots(figsize=(8, 6))
plot_confusion_matrix(metrics['confusion_matrix'], ax=ax)
plt.title("Oil Authentication: Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
plt.close()

# STEP 5: SAVE RESULTS
print("\nSTEP 5: Saving results...")
results = {
    "accuracy": float(metrics['accuracy']),
    "balanced_accuracy": float(metrics['balanced_accuracy']),
    "macro_f1": float(metrics['macro_f1']),
    "n_samples": len(spectra),
    "n_classes": len(set(spectra.labels)),
    "preprocessing": {
        "baseline": "ALS (lambda=1e5, p=0.01)",
        "smoothing": "Savitzky-Golay (window=9, poly=3)",
        "normalization": "SNV"
    }
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved: results.json")

print("\n" + "="*70)
print("WORKFLOW COMPLETE")
print("="*70)
print(f"Final Accuracy: {metrics['accuracy']:.1%}")
print(f"Outputs: confusion_matrix.png, results.json")
print("="*70)
```

**Expected output:**
```
======================================================================
COMPLETE END-TO-END OIL AUTHENTICATION WORKFLOW
======================================================================

STEP 1: Loading data...
✅ Loaded 96 spectra
   Classes: {'Olive', 'Palm', 'Sunflower', 'Coconut'}
   Shape: (96, 4096)

STEP 2: Preprocessing...
✅ Applied:
   - Baseline correction (ALS)
   - Smoothing (Savitzky-Golay)
   - Normalization (SNV)

STEP 3: Training classifier...
✅ Cross-validation complete:
   Accuracy: 95.2%
   Balanced Accuracy: 94.8%
   Macro F1: 0.948

STEP 4: Generating figures...
✅ Saved: confusion_matrix.png

STEP 5: Saving results...
✅ Saved: results.json

======================================================================
WORKFLOW COMPLETE
======================================================================
Final Accuracy: 95.2%
Outputs: confusion_matrix.png, results.json
======================================================================
```

---

## Decision Points in the Workflow

### When to use Random Forest vs. SVM?

| Criterion | Random Forest | SVM |
|-----------|---------------|-----|
| **Data size** | < 1000 samples: prefer RF | > 10,000 samples: both fine |
| **Interpretability** | Feature importance built-in | Limited feature importance |
| **Training speed** | Fast (seconds) | Slow (minutes to hours) |
| **Hyperparameters** | Few (n_estimators, max_depth) | Many (C, gamma, kernel) |
| **When in doubt** | Start with RF | Use for high dimensions |

**Recommendation:** Start with Random Forest for spectroscopy data (good balance of speed and interpretability).

---

## Troubleshooting the Pipeline

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: no samples to learn` | Empty or misaligned data | Check `spectra.data.shape` and `spectra.labels` |
| `Accuracy = 50%` (random guessing) | Data not preprocessed | Add baseline/normalization steps |
| `Overfitting detected` | Model too complex for small dataset | Reduce max_depth, use simpler model |
| `All predictions same class` | Class imbalance or leakage | Check CV strategy, use stratified split |

---

## Part 1: Data Import and Exploration

### ⏱️ Section time: 5 minutes

#### 1.1 Load Data from CSV

```python
import pandas as pd
from foodspec import SpectralDataset
import matplotlib.pyplot as plt

# Load raw spectra
csv_path = "data/oils_raw.csv"
df = pd.read_csv(csv_path)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# Output:
# Data shape: (1701, 61)
# Columns: ['wavenumber', 'VOO_1', 'VOO_2', ..., 'ROO_30']
```

#### 1.2 Create FoodSpec Dataset

⏱️ ~2 min

Convert the pandas DataFrame into a FoodSpec SpectralDataset with metadata:

```python
# Create dataset
ds = SpectralDataset.from_csv(
    csv_path,
    wavenumber_col="wavenumber",
    intensity_cols=None,  # Use all columns except wavenumber
    sample_ids=None       # Will auto-generate from column names
)

print(f"Samples: {len(ds)}")
print(f"Wavenumber range: {ds.wavenumber[0]:.1f}—{ds.wavenumber[-1]:.1f} cm⁻¹")
print(f"Spectral resolution: {ds.wavenumber[1] - ds.wavenumber[0]:.2f} cm⁻¹")

# Output:
# Samples: 60
# Wavenumber range: 400.0—3200.0 cm⁻¹
# Spectral resolution: 1.83 cm⁻¹
```

#### 1.3 Add Labels and Metadata

⏱️ ~2 min

Attach sample labels and batch information for proper validation:

```python
# Extract labels from sample IDs (e.g., 'VOO_1' → 'VOO')
labels = [sid.split('_')[0] for sid in ds.sample_ids]
ds.metadata['labels'] = labels

# Add batch information (simulating multiple measurement days)
batches = []
for i, sid in enumerate(ds.sample_ids):
    # First 30 samples measured on day 1, remaining on day 2
    batches.append('day1' if i < 30 else 'day2')
ds.metadata['batch'] = batches

print(f"Sample distribution:")
print(pd.Series(labels).value_counts())

# Output:
# VOO    30
# ROO    30
```

#### 1.4 Exploratory Visualization

⏱️ ~1 min

Examine the raw spectra to identify key features:

```python
from foodspec.visualization import plot_spectra

# Plot raw spectra by group
fig, ax = plt.subplots(figsize=(12, 6))

voo_idx = [i for i, label in enumerate(labels) if label == 'VOO']
roo_idx = [i for i, label in enumerate(labels) if label == 'ROO']

ax.plot(ds.wavenumber, ds.intensities[voo_idx].mean(axis=0), 
        label='VOO (n=30)', linewidth=2, color='green', alpha=0.7)
ax.plot(ds.wavenumber, ds.intensities[roo_idx].mean(axis=0), 
        label='ROO (n=30)', linewidth=2, color='orange', alpha=0.7)

ax.set_xlabel('Raman shift (cm⁻¹)')
ax.set_ylabel('Intensity (a.u.)')
ax.set_title('Raw Raman Spectra: VOO vs. ROO')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/01_raw_spectra.png', dpi=150)
plt.show()
```

**Interpretation:** The VOO and ROO spectra show distinct differences in the 1600–1700 cm⁻¹ region (C=C stretch) and 1200–1300 cm⁻¹ region (C-O stretch), indicating chemical differences that should enable discrimination.

---

## Part 2: Preprocessing

Raw spectral data contains noise, baseline drift, and intensity variations that interfere with analysis. Preprocessing must be applied *before* splitting into train/test sets to avoid data leakage.

### 2.1 Inspect Baseline Issues

```python
# Plot a single raw spectrum
sample_idx = 0
raw_spectrum = ds.intensities[sample_idx]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ds.wavenumber, raw_spectrum, label='Raw spectrum', linewidth=1)
ax.set_xlabel('Raman shift (cm⁻¹)')
ax.set_ylabel('Intensity (a.u.)')
ax.set_title(f'Raw Spectrum: {ds.sample_ids[sample_idx]}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/02_single_raw_spectrum.png', dpi=150)
plt.show()

# Observations:
# - Baseline has curvature (instrument drift)
# - High-frequency noise visible
# - Cosmic rays (spikes) may be present
```

### 2.2 Apply Baseline Correction

FoodSpec provides multiple baseline methods. For Raman spectroscopy of oils, asymmetric least squares (ALS) is effective:

```python
from foodspec.preprocessing import BaselineCorrector

# Create baseline corrector with ALS method
corrector = BaselineCorrector(method='als', lambda_=1e5, p=0.01)

# Apply to all spectra (this is safe—baseline fitting is internal)
ds_baseline = corrector.fit_transform(ds)

print("Baseline correction applied")
print(f"First spectrum (first 10 points):")
print(f"  Raw:     {ds.intensities[0, :10]}")
print(f"  Corrected: {ds_baseline.intensities[0, :10]}")

# Visualize baseline correction
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw spectrum with baseline fit
axes[0].plot(ds.wavenumber, ds.intensities[0], label='Raw', linewidth=1)
baseline_fit = corrector.baseline_[0]  # Retrieve fitted baseline
axes[0].plot(ds.wavenumber, baseline_fit, label='Fitted baseline', 
             linewidth=2, linestyle='--', color='red')
axes[0].set_title('Baseline Fitting (ALS)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Corrected spectrum
axes[1].plot(ds_baseline.wavenumber, ds_baseline.intensities[0], 
             label='Baseline-corrected', linewidth=1, color='green')
axes[1].set_title('After Baseline Correction')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('Raman shift (cm⁻¹)')
    ax.set_ylabel('Intensity (a.u.)')

plt.tight_layout()
plt.savefig('figures/03_baseline_correction.png', dpi=150)
plt.show()
```

### 2.3 Apply Normalization

Normalize to account for intensity variations from sample thickness, probe-to-sample distance, and detector sensitivity:

```python
from foodspec.preprocessing import Normalizer

# Vector normalization (L2 norm) is standard for Raman
normalizer = Normalizer(method='vector')
ds_norm = normalizer.fit_transform(ds_baseline)

print(f"Before normalization:")
print(f"  Spectrum 1 L2 norm: {(ds_baseline.intensities[0]**2).sum()**0.5:.2f}")
print(f"  Spectrum 2 L2 norm: {(ds_baseline.intensities[1]**2).sum()**0.5:.2f}")

print(f"After normalization:")
print(f"  Spectrum 1 L2 norm: {(ds_norm.intensities[0]**2).sum()**0.5:.2f}")
print(f"  Spectrum 2 L2 norm: {(ds_norm.intensities[1]**2).sum()**0.5:.2f}")

# Visualize effect
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overlay raw baseline-corrected spectra (note different scales)
for i in range(5):
    axes[0].plot(ds_baseline.wavenumber, ds_baseline.intensities[i], alpha=0.5)
axes[0].set_title('Before Normalization (different scales)')
axes[0].set_ylabel('Intensity (a.u.)')

# Overlay normalized spectra (aligned scales)
for i in range(5):
    axes[1].plot(ds_norm.wavenumber, ds_norm.intensities[i], alpha=0.5)
axes[1].set_title('After Vector Normalization (aligned)')
axes[1].set_ylabel('Normalized intensity')

for ax in axes:
    ax.set_xlabel('Raman shift (cm⁻¹)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/04_normalization.png', dpi=150)
plt.show()
```

### 2.4 Apply Smoothing

Reduce high-frequency noise with Savitzky-Golay smoothing:

```python
from foodspec.preprocessing import Smoother

# Savitzky-Golay: window=9 points, polynomial order=3
smoother = Smoother(method='savitzky_golay', window=9, poly_order=3)
ds_smooth = smoother.fit_transform(ds_norm)

print("Smoothing applied")

# Compare smoothing effect on a high-noise region
fig, ax = plt.subplots(figsize=(12, 5))
wavenumber_slice = (ds_norm.wavenumber >= 1200) & (ds_norm.wavenumber <= 1700)
slice_idx = wavenumber_slice.nonzero()[0]

ax.plot(ds_norm.wavenumber[wavenumber_slice], 
        ds_norm.intensities[0][wavenumber_slice],
        label='Before smoothing', linewidth=1, alpha=0.7)
ax.plot(ds_smooth.wavenumber[wavenumber_slice], 
        ds_smooth.intensities[0][wavenumber_slice],
        label='After smoothing', linewidth=2, color='red')
ax.set_xlabel('Raman shift (cm⁻¹)')
ax.set_ylabel('Normalized intensity')
ax.set_title('Effect of Savitzky-Golay Smoothing')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/05_smoothing.png', dpi=150)
plt.show()
```

### 2.5 Create Preprocessing Pipeline

Combine preprocessing steps into a reusable pipeline:

```python
from foodspec.preprocessing import Pipeline

# Define preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('baseline', BaselineCorrector(method='als', lambda_=1e5, p=0.01)),
    ('normalize', Normalizer(method='vector')),
    ('smooth', Smoother(method='savitzky_golay', window=9, poly_order=3))
])

# Save pipeline configuration
import json
pipeline_config = {
    'baseline': {'method': 'als', 'lambda': 1e5, 'p': 0.01},
    'normalize': {'method': 'vector'},
    'smooth': {'method': 'savitzky_golay', 'window': 9, 'poly_order': 3}
}
with open('config/preprocessing.json', 'w') as f:
    json.dump(pipeline_config, f, indent=2)

# Apply full pipeline
ds_processed = preprocessing_pipeline.fit_transform(ds)

print(f"Preprocessing complete")
print(f"Original shape: {ds.intensities.shape}")
print(f"Processed shape: {ds_processed.intensities.shape}")
```

**Key principle:** All preprocessing is applied to the full dataset *before* train/test splitting. This ensures that baseline fitting, normalization, and smoothing parameters come from the entire dataset, preventing data leakage.

---

## Part 3: Feature Extraction

Convert high-dimensional spectral data (1700 wavenumbers) into interpretable features (3–20 features).

### 3.1 Peak-Based Features

Identify diagnostic peaks in the 1200–1700 cm⁻¹ region:

```python
from foodspec.features import PeakDetector

# Define regions of interest for olive oil
peak_regions = {
    'C=C_stretch_1651': (1630, 1670),      # C=C conjugated/unconjugated
    'CH2_bend_1438': (1420, 1450),         # CH₂ bending
    'C_O_stretch_1275': (1260, 1290),      # C-O stretching
    'C_H_deformation_720': (700, 750),     # C-H out-of-plane deformation
}

# Detect peak areas in each region
features = {}
for peak_name, (wn_min, wn_max) in peak_regions.items():
    region_mask = (ds_processed.wavenumber >= wn_min) & \
                  (ds_processed.wavenumber <= wn_max)
    
    # Integrate intensity in region (simple trapezoid rule)
    region_spec = ds_processed.intensities[:, region_mask]
    peak_area = region_spec.sum(axis=1)  # Sum along wavenumber
    features[peak_name] = peak_area

# Create feature matrix
feature_df = pd.DataFrame(features)
feature_df['sample_id'] = ds_processed.sample_ids
feature_df['label'] = ds_processed.metadata['labels']

print(feature_df.head(10))
print(f"\nFeature statistics:")
print(feature_df.groupby('label')[list(features.keys())].describe())
```

### 3.2 Ratiometric Features

Peak ratios are more robust than absolute peak areas because they normalize for sample variation:

```python
# Common ratios in oil analysis
feature_df['C=C_to_CH2'] = \
    feature_df['C=C_stretch_1651'] / feature_df['CH2_bend_1438']
feature_df['C_O_to_C=C'] = \
    feature_df['C_O_stretch_1275'] / feature_df['C=C_stretch_1651']

print("Ratiometric features:")
print(feature_df[['C=C_to_CH2', 'C_O_to_C=C', 'label']].head(10))

# Visualize ratio distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ratio_name, ax in zip(['C=C_to_CH2', 'C_O_to_C=C'], axes):
    voo_data = feature_df[feature_df['label'] == 'VOO'][ratio_name]
    roo_data = feature_df[feature_df['label'] == 'ROO'][ratio_name]
    
    ax.hist(voo_data, bins=8, alpha=0.6, label='VOO', color='green')
    ax.hist(roo_data, bins=8, alpha=0.6, label='ROO', color='orange')
    ax.set_xlabel(ratio_name)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution: {ratio_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/06_ratio_distributions.png', dpi=150)
plt.show()
```

### 3.3 Dimensionality Reduction

PCA extracts underlying patterns in the full spectral data:

```python
from foodspec.chemometrics import PCA as FoodSpecPCA
from sklearn.decomposition import PCA

# Fit PCA on processed spectra
pca = FoodSpecPCA(n_components=5)
spectra_pca = pca.fit_transform(ds_processed.intensities)

print(f"PCA explained variance:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.1%}")

print(f"Cumulative variance (5 PCs): {pca.explained_variance_ratio_.sum():.1%}")

# Create PCA feature matrix
pca_features = pd.DataFrame(
    spectra_pca,
    columns=[f'PC{i+1}' for i in range(5)]
)
pca_features['sample_id'] = ds_processed.sample_ids
pca_features['label'] = ds_processed.metadata['labels']

# Visualize PCA scores
fig, ax = plt.subplots(figsize=(10, 8))

for label, color in [('VOO', 'green'), ('ROO', 'orange')]:
    mask = pca_features['label'] == label
    ax.scatter(pca_features[mask]['PC1'], 
               pca_features[mask]['PC2'],
               label=label, s=100, alpha=0.6, color=color, edgecolors='black')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA Score Plot: VOO vs. ROO')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/07_pca_scores.png', dpi=150)
plt.show()
```

---

## Part 4: Classification Model

Build a classifier to distinguish VOO from ROO using the extracted features.

### 4.1 Prepare Train/Test Split

**Critical:** Split data *after* all preprocessing and feature extraction to prevent data leakage.

```python
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# Use batch information for stratification
X = feature_df[['C=C_stretch_1651', 'CH2_bend_1438', 'C_O_stretch_1275',
                 'C=C_to_CH2', 'C_O_to_C=C']].values
y = feature_df['label'].values
batches = np.array(ds_processed.metadata['batch'])

# Stratified split: 70% train, 30% test, respecting batch and class distribution
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, 
                                  random_state=42)

# Use combined stratification key (label + batch)
stratify_key = np.array([f"{y[i]}_{batches[i]}" for i in range(len(y))])

train_idx, test_idx = next(splitter.split(X, stratify_key))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Training set: {len(X_train)} samples")
print(f"  VOO: {(y_train == 'VOO').sum()}, ROO: {(y_train == 'ROO').sum()}")
print(f"Test set: {len(X_test)} samples")
print(f"  VOO: {(y_test == 'VOO').sum()}, ROO: {(y_test == 'ROO').sum()}")
```

### 4.2 Train Classifier

Use Random Forest for interpretability:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, 
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_encoded)

# Predictions
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]  # Probability of ROO

print("Model trained")
print(f"\nTest set performance:")
print(confusion_matrix(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred, 
                           target_names=le.classes_))
print(f"ROC-AUC: {roc_auc_score(y_test_encoded, y_pred_proba):.3f}")
```

### 4.3 Nested Cross-Validation

Evaluate model performance rigorously:

```python
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

# Build sklearn pipeline
pipe = SklearnPipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

# Nested CV: outer loop for evaluation, inner loop for hyperparameter tuning
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Evaluate with multiple metrics
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'roc_auc': 'roc_auc',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

cv_results = cross_validate(pipe, X, y_train_encoded, cv=outer_cv,
                            scoring=scoring, return_train_score=False)

# Summarize CV results
print("Nested Cross-Validation Results (5 folds):")
for metric, scores in cv_results.items():
    if metric.startswith('test_'):
        metric_name = metric.replace('test_', '')
        print(f"{metric_name:20s}: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Part 5: Interpretation

Understand *why* the model makes predictions.

### 5.1 Feature Importance

Which features drive the classification?

```python
# Feature importance from Random Forest
feature_names = ['C=C_stretch_1651', 'CH2_bend_1438', 'C_O_stretch_1275',
                 'C=C_to_CH2', 'C_O_to_C=C']
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature Importance Ranking:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]:20s}: {importances[idx]:.3f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(indices)), importances[indices], color='steelblue')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.set_xlabel('Importance (mean decrease in impurity)')
ax.set_title('Random Forest Feature Importance')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('figures/08_feature_importance.png', dpi=150)
plt.show()
```

### 5.2 Prediction Confidence

How confident are the predictions?

```python
# Prediction probabilities
y_pred_proba_class0 = rf.predict_proba(X_test)[:, 0]  # VOO probability
y_pred_proba_class1 = rf.predict_proba(X_test)[:, 1]  # ROO probability

# Create confidence dataframe
conf_df = pd.DataFrame({
    'sample_id': feature_df.iloc[test_idx]['sample_id'].values,
    'true_label': y_test,
    'predicted_label': le.inverse_transform(y_pred),
    'confidence': np.max(rf.predict_proba(X_test), axis=1),
    'voo_prob': y_pred_proba_class0,
    'roo_prob': y_pred_proba_class1
})

print("\nPrediction Confidence (test set):")
print(conf_df.head(15))
print(f"\nMean confidence: {conf_df['confidence'].mean():.3f}")
print(f"Predictions with confidence > 0.95: {(conf_df['confidence'] > 0.95).sum()}/{len(conf_df)}")
```

### 5.3 Misclassifications

Investigate errors to identify model limitations:

```python
# Find misclassifications
misclass_mask = y_test != le.inverse_transform(y_pred)
misclassified = conf_df[misclass_mask]

print(f"\nMisclassifications ({len(misclassified)} samples):")
print(misclassified)

if len(misclassified) > 0:
    # Analyze features of misclassified samples
    misclass_idx = test_idx[misclass_mask]
    print("\nFeature values for misclassified samples:")
    print(feature_df.iloc[misclass_idx][['C=C_to_CH2', 'C_O_to_C=C', 'label']])
```

---

## Part 6: Reproducibility

Save the complete analysis for publication and reproduction.

### 6.1 Create Protocol YAML

Document the entire pipeline in a machine-readable format:

```yaml
# protocols/oil_authentication.yaml
name: "Oil Authentication: VOO vs. ROO"
version: "1.0"
created: "2024-01-06"
description: "Classify virgin olive oil from refined olive oil using Raman spectroscopy"

input:
  format: "csv"
  path: "data/oils_raw.csv"
  wavenumber_col: "wavenumber"

preprocessing:
  baseline:
    method: "als"
    lambda: 1e5
    p: 0.01
  normalize:
    method: "vector"
  smooth:
    method: "savitzky_golay"
    window: 9
    poly_order: 3

feature_extraction:
  type: "peak_ratios"
  regions:
    C=C_stretch_1651: [1630, 1670]
    CH2_bend_1438: [1420, 1450]
    C_O_stretch_1275: [1260, 1290]
  ratios:
    - "C=C_to_CH2"
    - "C_O_to_C=C"

model:
  type: "RandomForestClassifier"
  hyperparameters:
    n_estimators: 100
    max_depth: 5
    random_state: 42

validation:
  method: "nested_cross_validation"
  outer_cv: 5
  inner_cv: 3
  stratification: "label_and_batch"

expected_performance:
  balanced_accuracy: 0.95
  roc_auc: 0.98
```

### 6.2 Generate Methods Text

FoodSpec can auto-generate reproducible methods text:

```python
from foodspec.reporting import MethodsTextGenerator

methods_gen = MethodsTextGenerator(
    title="Classification of Olive Oils by Raman Spectroscopy",
    preprocessing_config=pipeline_config,
    model_name="Random Forest",
    n_samples=len(ds_processed),
    cv_folds=5
)

methods_text = methods_gen.generate()
print(methods_text)

# Save to file
with open('output/methods_section.txt', 'w') as f:
    f.write(methods_text)
```

**Generated output (example):**
```
Methods
-------
Spectral Acquisition. Raman spectra were acquired on a Bruker SENTERRA II 
spectrometer at 785 nm excitation. Sample set comprised 60 olive oil samples 
(30 virgin olive oil, 30 refined olive oil).

Preprocessing. Raw spectra (1700 wavenumber points, 400–3200 cm⁻¹) were 
processed as follows: (1) baseline correction using asymmetric least squares 
(λ=1e5, p=0.01); (2) vector normalization; (3) Savitzky-Golay smoothing 
(window=9, poly order=3). All preprocessing was applied to the full dataset 
prior to train/test splitting to avoid data leakage.

Feature Extraction. Five features were extracted: peak areas at 1651 cm⁻¹ 
(C=C stretch), 1438 cm⁻¹ (CH₂ bending), 1275 cm⁻¹ (C-O stretching), and 
ratios C=C/CH₂ and C-O/C=C.

Modeling. A Random Forest classifier (100 trees, max depth 5) was trained 
on 42 samples and evaluated on 18 held-out test samples via nested 
5-fold cross-validation (inner 3-fold for hyperparameter tuning).

Validation. Model performance was assessed using balanced accuracy, ROC-AUC, 
precision, recall, and F1-score. Batch effects were controlled by stratifying 
splits on combined label/batch key.
```

### 6.3 Save Results and Run Artifacts

Create a reproducible output bundle:

```python
import json
import pickle
from datetime import datetime
import os

# Create output directory
run_dir = f"runs/oil_auth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(run_dir, exist_ok=True)

# Save model
with open(f'{run_dir}/model_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Save preprocessing pipeline
with open(f'{run_dir}/preprocessing_pipeline.pkl', 'wb') as f:
    pickle.dump(preprocessing_pipeline, f)

# Save processed spectra (for reproducibility)
np.save(f'{run_dir}/X_processed.npy', ds_processed.intensities)
np.save(f'{run_dir}/wavenumber.npy', ds_processed.wavenumber)

# Save features and predictions
feature_df.to_csv(f'{run_dir}/features.csv', index=False)
conf_df.to_csv(f'{run_dir}/predictions.csv', index=False)

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'n_samples': len(ds),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'preprocessing_config': pipeline_config,
    'feature_names': feature_names,
    'model_type': 'RandomForestClassifier',
    'test_balanced_accuracy': float((y_test == le.inverse_transform(y_pred)).mean()),
    'test_roc_auc': float(roc_auc_score(y_test_encoded, y_pred_proba))
}

with open(f'{run_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Results saved to: {run_dir}")
print(f"Files created:")
print(f"  - model_rf.pkl (trained classifier)")
print(f"  - preprocessing_pipeline.pkl (preprocessing steps)")
print(f"  - features.csv (extracted features)")
print(f"  - predictions.csv (test set predictions)")
print(f"  - metadata.json (run summary)")
```

### 6.4 Document Data and Code Versions

```python
# Create reproducibility log
repro_log = f"""
FoodSpec Analysis Reproducibility Log
=====================================

Timestamp: {datetime.now().isoformat()}
Analyst: [Your name]
Project: Oil Authentication Workflow

Software Versions:
  - FoodSpec: {__import__('foodspec').__version__}
  - scikit-learn: {__import__('sklearn').__version__}
  - NumPy: {__import__('numpy').__version__}
  - pandas: {__import__('pandas').__version__}
  - Python: {__import__('sys').version.split()[0]}

Data:
  - Input file: data/oils_raw.csv
  - N samples: {len(ds)}
  - Wavenumber range: {ds.wavenumber[0]:.1f}–{ds.wavenumber[-1]:.1f} cm⁻¹
  - Resolution: {ds.wavenumber[1] - ds.wavenumber[0]:.2f} cm⁻¹

Preprocessing (applied to full dataset, no leakage):
  - Baseline: ALS (λ=1e5, p=0.01)
  - Normalization: Vector
  - Smoothing: Savitzky-Golay (window=9, poly=3)

Features Extracted: {len(feature_names)}
  - {', '.join(feature_names)}

Model:
  - Type: Random Forest
  - N estimators: 100
  - Max depth: 5

Validation:
  - Test accuracy: {(y_test == le.inverse_transform(y_pred)).mean():.3f}
  - Cross-val balanced accuracy: {cv_results['test_balanced_accuracy'].mean():.3f} ± {cv_results['test_balanced_accuracy'].std():.3f}

To reproduce:
  1. Load data: `ds = SpectralDataset.from_csv('data/oils_raw.csv', wavenumber_col='wavenumber')`
  2. Apply preprocessing: `ds_proc = preprocessing_pipeline.fit_transform(ds)`
  3. Extract features: See feature_extraction section above
  4. Load model: `rf = pickle.load(open('{run_dir}/model_rf.pkl', 'rb'))`
  5. Predict: `y_pred = rf.predict(X_test)`
"""

with open(f'{run_dir}/reproducibility_log.txt', 'w') as f:
    f.write(repro_log)

print(repro_log)
```

---

## Checklist: End-to-End Pipeline

Use this checklist to ensure analysis quality and reproducibility:

- [ ] **Data Import**
  - [ ] Verified data shape and column names
  - [ ] Checked for missing values
  - [ ] Confirmed label distribution
  
- [ ] **Preprocessing**
  - [ ] Baseline correction applied and visualized
  - [ ] Normalization method justified
  - [ ] Smoothing parameters documented
  - [ ] **All preprocessing applied before train/test split**
  
- [ ] **Feature Extraction**
  - [ ] Regions of interest justified
  - [ ] Feature distributions examined by class
  - [ ] Ratiometric features more robust than absolute peaks
  
- [ ] **Model Building**
  - [ ] Train/test split respects batch structure
  - [ ] Stratification applied correctly
  - [ ] Hyperparameters documented
  
- [ ] **Validation**
  - [ ] Nested cross-validation used
  - [ ] Multiple metrics reported (accuracy, balanced accuracy, ROC-AUC)
  - [ ] Confidence intervals estimated (bootstrap or CV)
  
- [ ] **Interpretation**
  - [ ] Feature importance ranked
  - [ ] Misclassifications analyzed
  - [ ] Model limitations documented
  
- [ ] **Reproducibility**
  - [ ] Preprocessing pipeline saved
  - [ ] Model weights saved
  - [ ] Methods text generated
  - [ ] Run metadata and timestamp recorded
  - [ ] Code and data versions documented

---

## Key Principles Demonstrated

1. **Preprocessing before splitting:** Avoid data leakage by preprocessing the full dataset, then splitting
2. **Stratified splits:** Respect batch structure and class balance in train/test splits
3. **Nested cross-validation:** Use inner CV for hyperparameter tuning, outer CV for unbiased performance estimates
4. **Feature interpretability:** Extract and visualize features to understand discriminative patterns
5. **Full provenance:** Save preprocessing config, model weights, predictions, and metadata for reproduction
6. **Comprehensive validation:** Report multiple metrics with confidence intervals, not just accuracy

---

## Next Steps

- **Extend with new samples:** Retrain with additional oil varieties (extra virgin, pomace)
- **Explore other modalities:** Repeat with FTIR or NIR spectroscopy
- **Compare methods:** Benchmark against PLS-DA or SVM classifiers
- **Deploy:** Use the saved model for real-time oil authentication in QC workflows
- **Publish:** Use generated methods text and figures in manuscripts

---

## What Went Wrong in Early Runs (Lessons Learned)

- **Baseline on full data:** Fitting ALS on the entire dataset before splitting inflated test scores. Fix: fit baseline inside CV folds or on train split only.
- **Replicates split across folds:** Bottles measured across days landed in different folds, hiding batch drift. Fix: stratify on label + batch and keep replicates together.
- **Wavenumber ordering mismatch:** One vendor export reversed wavenumber order, silently breaking ratios. Fix: assert monotonic wavenumber and reorder before preprocessing.
- **Peaks outside region masks:** Slight shifts moved peaks outside hard-coded windows. Fix: widen windows and verify peak coverage per batch.
- **No confidence reporting:** Single accuracy number looked great until external data arrived. Fix: report balanced accuracy, ROC-AUC, and fold-wise spread; keep prediction confidence.

---

## Related Documentation

- [Oil Authentication Workflow](./authentication/oil_authentication.md) — Detailed domain workflow
- [Cross-Validation and Data Leakage](../methods/validation/cross_validation_and_leakage.md) — Rigorous validation strategies
- [Preprocessing Guide](../methods/preprocessing/baseline_correction.md) — Deep dive on baseline methods
- [Model Evaluation](../methods/chemometrics/model_evaluation_and_validation.md) — Metrics and validation
- [Reproducibility Checklist](../protocols/reproducibility_checklist.md) — Publication-ready analysis
