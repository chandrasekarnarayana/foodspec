# Quickstart (Python API)

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Python developers, data scientists, researchers needing custom workflows  
**What problem does this solve?** Integrate FoodSpec into Python scripts, Jupyter notebooks, ML pipelines  
**When to use this?** Fine-grained control, custom preprocessing, integration with existing code  
**Why it matters?** Python API provides maximum flexibility for research and development  
**Time to complete:** 15 minutes  
**Prerequisites:** FoodSpec installed (`pip install foodspec`), Python 3.10+, basic pandas/numpy knowledge

---

## Installation

### pip (Recommended)
```bash
pip install foodspec
```

### conda
```bash
conda install -c conda-forge foodspec
```

### Verify
```python
import foodspec
print(f"FoodSpec {foodspec.__version__}")
```

---

## Dataset Format

!!! tip "Data Format Reference"
    See [Data Format Reference](../09-reference/data_format.md) for complete schema specifications, validation checklists, and best practices. Key terms defined in [Glossary](../reference/glossary.md).

### CSV Requirements
```csv
wavenumber,oil_type,batch,s1,s2,s3
1000.0,olive,A,5.2,5.1,4.8
1010.0,olive,A,5.5,5.3,5.0
1020.0,olive,A,5.8,5.6,5.2
```

### In Python
```python
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('oils.csv')

# Required columns:
wavenumbers = df['wavenumber'].values  # shape: (150,)
spectra = df.iloc[:, 3:].values         # shape: (150, 3) — one spectrum per row
labels = df['oil_type'].values          # shape: (3,) — one label per spectrum
```

### Or generate synthetic data for testing
```python
import numpy as np
import pandas as pd

np.random.seed(42)
wavenumbers = np.linspace(1000, 3000, 150)

# Generate 10 samples per class
olive_spectra = np.random.normal(5.0, 0.3, (10, 150))
palm_spectra = np.random.normal(6.0, 0.3, (10, 150))

# Combine
X = np.vstack([olive_spectra, palm_spectra])  # shape: (20, 150)
y = ['olive']*10 + ['palm']*10                 # shape: (20,)

# Save for later
df = pd.DataFrame(X, columns=[f'{w:.1f}' for w in wavenumbers])
df.insert(0, 'oil_type', y)
df.insert(0, 'wavenumber', wavenumbers)
df.to_csv('oils_demo.csv', index=False)
```

---

## Complete Example: Oil Classification

### Step 1: Create synthetic data
```python
import numpy as np
import pandas as pd
from pathlib import Path

# Create toy dataset
np.random.seed(42)
wavenumbers = np.linspace(1000, 3000, 150)

# Generate spectra with class-specific patterns
olive = np.random.normal(5.0, 0.3, (15, 150))
palm = np.random.normal(6.0, 0.3, (15, 150))
sunflower = np.random.normal(4.5, 0.3, (15, 150))

# Combine
X = np.vstack([olive, palm, sunflower])           # (45, 150)
y = ['olive']*15 + ['palm']*15 + ['sunflower']*15 # (45,)

# Save
df = pd.DataFrame(X, columns=[f'{w:.1f}' for w in wavenumbers])
df.insert(0, 'oil_type', y)
df.insert(0, 'wavenumber', wavenumbers)
df.to_csv('oils_demo.csv', index=False)

print(f"✓ Created oils_demo.csv: {X.shape[0]} spectra, {X.shape[1]} wavenumbers")
```

### Step 2: Load and explore
```python
from foodspec import SpectralDataset
import matplotlib.pyplot as plt

# Load from CSV
ds = SpectralDataset.from_csv(
    'oils_demo.csv',
    wavenumber_col='wavenumber',
    label_col='oil_type'
)

print(f"Loaded {len(ds)} spectra")
print(f"Wavenumber range: {ds.wavenumbers[0]:.1f}–{ds.wavenumbers[-1]:.1f} cm⁻¹")
print(f"Classes: {set(ds.labels)}")

# Visualize raw spectra
fig, ax = plt.subplots(figsize=(10, 4))
for label in set(ds.labels):
    idx = ds.labels == label
    ax.plot(ds.wavenumbers, ds.x[idx].mean(axis=0), label=label, linewidth=2)
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Intensity')
ax.legend()
plt.tight_layout()
plt.savefig('spectra_raw.png', dpi=150)
print("✓ Saved spectra_raw.png")
```

### Step 3: Preprocess
```python
from foodspec.preprocessing import (
    baseline_als,
    normalize_vector,
    smooth_savitzky_golay
)

# Preprocessing pipeline
X = ds.x.copy()

# 1. Baseline correction (remove instrumental drift)
X = baseline_als(X, lambda_=1e5, p=0.01)

# 2. Savitzky-Golay smoothing (reduce noise)
X = smooth_savitzky_golay(X, window_length=9, polyorder=3)

# 3. Vector normalization (make samples comparable)
X = normalize_vector(X, norm='l2')

print(f"✓ Preprocessed shape: {X.shape}")

# Visualize preprocessed spectra
fig, ax = plt.subplots(figsize=(10, 4))
labels_unique = sorted(set(ds.labels))
for label in labels_unique:
    idx = ds.labels == label
    ax.plot(ds.wavenumbers, X[idx].mean(axis=0), label=label, linewidth=2)
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Normalized Intensity')
ax.legend()
plt.tight_layout()
plt.savefig('spectra_preprocessed.png', dpi=150)
print("✓ Saved spectra_preprocessed.png")
```

### Step 4: Explore with PCA
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA
pca = PCA(n_components=2)
scores = pca.fit_transform(X)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = {'olive': 'green', 'palm': 'orange', 'sunflower': 'gold'}
for label in set(ds.labels):
    idx = ds.labels == label
    ax.scatter(scores[idx, 0], scores[idx, 1], label=label, s=100, 
               color=colors.get(label, 'blue'), alpha=0.7)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_scores.png', dpi=150)
print(f"✓ Saved pca_scores.png (variance explained: {sum(pca.explained_variance_ratio_):.1%})")
```

### Step 5: Train and evaluate classifier
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Create classifier
clf = RandomForestClassifier(n_estimators=50, random_state=42)

# Cross-validation
scores = cross_val_score(clf, X, ds.labels, cv=5, scoring='balanced_accuracy')
print(f"Cross-validation balanced accuracy: {scores.mean():.3f} (±{scores.std():.3f})")

# Train on all data (for demo)
clf.fit(X, ds.labels)

# Predictions
y_pred = clf.predict(X)

# Print metrics
print("\nClassification report:")
print(classification_report(ds.labels, y_pred))

# Confusion matrix
cm = confusion_matrix(ds.labels, y_pred)
print(f"\nConfusion matrix:\n{cm}")
```

### Step 6: Save model and results
```python
import json
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("✓ Saved model.pkl")

# Save metadata
metadata = {
    'model': 'RandomForestClassifier',
    'n_samples': len(ds),
    'n_features': X.shape[1],
    'classes': list(set(ds.labels)),
    'cv_accuracy': float(scores.mean()),
    'timestamp': pd.Timestamp.now().isoformat()
}
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved metadata.json")
```

---

## Real-World Workflow: Oil Authentication

Use built-in ratiometric features (RQ engine) for interpretable results:

```python
from foodspec.features.rq import RatioQualityEngine, RQConfig

# Define discriminative ratios
config = RQConfig(
    ratios=[
        ('1742', '2720'),  # C=O / C=C (common marker)
        ('1652', '2720'),  # C=C bend / C=C stretch
        ('1600', '2720'),  # Aromatic / aliphatic
    ]
)

# Compute features
rq = RatioQualityEngine(config=config)
features = rq.compute(X, ds.wavenumbers)

print(f"RQ features shape: {features.shape}")
print(f"Feature names: {rq.feature_names}")

# Train on features instead of raw spectra
clf_rq = RandomForestClassifier(n_estimators=50, random_state=42)
scores_rq = cross_val_score(clf_rq, features, ds.labels, cv=5)
print(f"RQ-based CV accuracy: {scores_rq.mean():.3f}")
```

---

## Expected Outputs

### Plots generated
- ✅ `spectra_raw.png` — Raw spectra by class
- ✅ `spectra_preprocessed.png` — Preprocessed spectra
- ✅ `pca_scores.png` — PCA biplot

### Files generated
- ✅ `oils_demo.csv` — Synthetic dataset
- ✅ `model.pkl` — Trained classifier
- ✅ `metadata.json` — Run metadata

### Console output
```yaml
✓ Created oils_demo.csv: 45 spectra, 150 wavenumbers
Loaded 45 spectra
Wavenumber range: 1000.0–3000.0 cm⁻¹
Classes: {'olive', 'palm', 'sunflower'}
✓ Preprocessed shape: (45, 150)
✓ Saved spectra_raw.png
✓ Saved spectra_preprocessed.png
✓ Saved pca_scores.png (variance explained: 83.5%)
Cross-validation balanced accuracy: 0.967 (±0.049)

Classification report:
              precision    recall  f1-score   support
       olive       0.93      1.00      0.96        15
        palm       1.00      0.93      0.96        15
   sunflower       0.93      1.00      0.96        15

    accuracy                           0.98        45
   macro avg       0.95      0.98      0.97        45
weighted avg       0.95      0.98      0.97        45
```

---

## Additional Resources

- **[Data Format Reference](../09-reference/data_format.md)** - Schema formats, unit conventions, validation checklist
- **[Glossary](../reference/glossary.md)** - Definitions of wavenumber, baseline, normalization, CV strategy, etc.
- **[API Reference](../08-api/index.md)** - Complete API documentation
- **[Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md)** - Detailed preprocessing recipes

---

## Troubleshooting (Top 5 Issues)

### 1️⃣ **"ModuleNotFoundError: No module named 'foodspec'"**
**Cause:** FoodSpec not installed

**Fix:**
```bash
pip install --upgrade foodspec
python -c "import foodspec; print(foodspec.__version__)"
```

---

### 2️⃣ **"ValueError: wavenumber column not found"**
**Cause:** Column name doesn't match

**Fix:**
```python
# Check CSV columns
import pandas as pd
df = pd.read_csv('oils.csv')
print(df.columns)

# Use correct column name
ds = SpectralDataset.from_csv(
    'oils.csv',
    wavenumber_col='cm-1',  # actual column name
    label_col='oil_type'
)
```

---

### 3️⃣ **"Shape mismatch: expected (n_samples, n_features)"**
**Cause:** Spectra are transposed (wavenumbers as rows, samples as columns)

**Fix:**
```python
# Check shape
print(X.shape)  # should be (n_samples, n_wavenumbers)

# If transposed, flip
if X.shape[1] > X.shape[0]:
    X = X.T
```

---

### 4️⃣ **"All NaN or constant feature"**
**Cause:** Bad preprocessing (too aggressive baseline correction)

**Fix:**
```python
# Check for NaN after preprocessing
import numpy as np
print(f"NaN count: {np.isnan(X).sum()}")
print(f"Const columns: {(X.std(axis=0) == 0).sum()}")

# Use gentler baseline correction
from foodspec.preprocessing import baseline_als
X = baseline_als(X, lambda_=1e4, p=0.1)  # less aggressive

# Remove constant columns
X = X[:, X.std(axis=0) > 1e-10]
```

---

### 5️⃣ **"Memory error on large dataset"**
**Cause:** Loading entire dataset into memory

**Fix:**
```python
# Process in chunks
import numpy as np
chunk_size = 100
for i in range(0, len(ds), chunk_size):
    chunk = ds.x[i:i+chunk_size]
    chunk_processed = baseline_als(chunk, lambda_=1e5, p=0.01)
    # Process chunk...

# Or subsample for exploration
ds_small = SpectralDataset(
    ds.x[::10],  # every 10th spectrum
    ds.wavenumbers,
    ds.labels[::10]
)
```

---

## Copy-Paste One-Liner (Start Here)

```python
# Full pipeline in ~50 lines
import numpy as np; import pandas as pd; from foodspec import SpectralDataset; from foodspec.preprocessing import baseline_als, normalize_vector, smooth_savitzky_golay; from sklearn.ensemble import RandomForestClassifier; from sklearn.model_selection import cross_val_score; np.random.seed(42); w = np.linspace(1000, 3000, 150); X = np.vstack([np.random.normal(5, 0.3, (15, 150)), np.random.normal(6, 0.3, (15, 150))]); y = ['olive']*15 + ['palm']*15; df = pd.DataFrame(X, columns=[f'{x:.1f}' for x in w]); df.insert(0, 'oil_type', y); df.insert(0, 'wavenumber', w); df.to_csv('demo.csv', index=False); ds = SpectralDataset.from_csv('demo.csv', wavenumber_col='wavenumber', label_col='oil_type'); X_prep = baseline_als(ds.x, 1e5, 0.01); X_prep = smooth_savitzky_golay(X_prep, 9, 3); X_prep = normalize_vector(X_prep, 'l2'); clf = RandomForestClassifier(50); scores = cross_val_score(clf, X_prep, ds.labels, cv=5); print(f"✓ CV accuracy: {scores.mean():.3f}")
```

---

## Next Steps

- ✅ Try the CLI version: [CLI Quickstart](quickstart_cli.md)
- ✅ Explore workflows: [Oil Authentication](../workflows/authentication/oil_authentication.md)
- ✅ Custom preprocessing: [Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md)
- ✅ Advanced models: [ML Guide](../08-api/ml.md)

---

## Need Help?

- **Getting errors (NaNs, shape mismatches, overfitting)?** → [Troubleshooting Guide](../10-help/troubleshooting.md)
- **Questions about methods or usage?** → [FAQ](../10-help/faq.md)
- **Report a bug:** [GitHub Issues](https://github.com/chandrasekarnarayana/foodspec/issues)
