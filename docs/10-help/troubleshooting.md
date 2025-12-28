# Troubleshooting

!!! info "Quick Help"
    This page provides solutions to common technical issues encountered when using FoodSpec. For conceptual questions, see the [FAQ](faq.md). For bug reports, see [Reporting Guidelines](../troubleshooting/reporting_guidelines.md).

---

## Installation Issues

### Problem: `pip install foodspec` fails

**Symptoms:**
```bash
ERROR: Could not find a version that satisfies the requirement foodspec
ERROR: No matching distribution found for foodspec
```

**Causes & Solutions:**

#### 1. Python Version Incompatibility

**Check your Python version:**
```bash
python --version
```

**Solution:** FoodSpec requires Python ≥3.8. Upgrade Python:
```bash
# Using conda
conda create -n foodspec python=3.10
conda activate foodspec
pip install foodspec

# Using pyenv
pyenv install 3.10.0
pyenv local 3.10.0
pip install foodspec
```

#### 2. Outdated pip

**Update pip:**
```bash
pip install --upgrade pip setuptools wheel
pip install foodspec
```

#### 3. Network/Firewall Issues

**Try alternative PyPI mirrors:**
```bash
# Use a specific PyPI mirror
pip install --index-url https://pypi.org/simple foodspec

# Install with verbose output to diagnose
pip install -v foodspec
```

#### 4. Package Name Confusion

**Verify the correct package name:**
```bash
# Correct
pip install foodspec

NOT: pip install food-spec, FoodSpec, foodspectra, etc.
```

---

### Problem: Import errors after installation

**Symptoms:**
```python
>>> import foodspec
ModuleNotFoundError: No module named 'foodspec'
```

**Causes & Solutions:**

#### 1. Multiple Python Environments

**Check which Python is active:**
```bash
which python
which pip
python -c "import sys; print(sys.executable)"
```

**Solution:** Ensure pip and python are from the same environment:
```bash
# Use python -m pip instead
python -m pip install foodspec

# Verify installation
python -c "import foodspec; print(foodspec.__version__)"
```

#### 2. Development Installation Not Linked

**If installing from source:**
```bash
# Editable install
cd /path/to/foodspec
pip install -e .

# Verify
python -c "import foodspec; print(foodspec.__file__)"
```

#### 3. PYTHONPATH Issues

**Check PYTHONPATH:**
```bash
echo $PYTHONPATH
```

**Solution:** Add FoodSpec to PYTHONPATH (if needed):
```bash
export PYTHONPATH="/path/to/foodspec/src:$PYTHONPATH"
```

---

## Data Loading Issues

- Verify file paths are correct relative to your working directory; prefer absolute paths when scripting.
- Confirm delimiters/headers match loader expectations (e.g., `wavenumber` column present for CSV/HDF5 helpers).
- For registry-driven runs, check that metadata tables point to existing files and have consistent sample IDs.

## Missing Dependencies

### Problem: Optional dependencies not installed

**Symptoms:**
```python
>>> from foodspec.visualization import plot_spectra
ImportError: matplotlib is required for visualization. Install with: pip install foodspec[viz]
```

**Solution:** Install optional dependency groups:

```bash
# Visualization (matplotlib, seaborn)
pip install foodspec[viz]

# Machine learning (scikit-learn, xgboost)
pip install foodspec[ml]

# All optional dependencies
pip install foodspec[all]

# Multiple groups
pip install foodspec[viz,ml]
```

**Available groups:**
- `viz`: Plotting and visualization
- `ml`: Machine learning models (RF, XGBoost)
- `notebooks`: Jupyter notebook support
- `dev`: Development tools (pytest, black, mypy)
- `docs`: Documentation building (mkdocs, mkdocstrings)
- `all`: All optional dependencies

---

### Problem: Conflicting dependency versions

**Symptoms:**
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
ERROR: foodspec requires numpy>=1.20, but you have numpy 1.19.5
```

**Solution 1: Upgrade conflicting packages**
```bash
pip install --upgrade numpy scipy scikit-learn
pip install foodspec
```

**Solution 2: Use a clean environment**
```bash
conda create -n foodspec-clean python=3.10
conda activate foodspec-clean
pip install foodspec
```

**Solution 3: Use conda for dependency management**
```bash
conda install -c conda-forge numpy scipy scikit-learn matplotlib
pip install foodspec
```

---

## Shape/Axis Mismatch Errors

### Problem: "Shapes do not match" during preprocessing

**Symptoms:**
```python
>>> from foodspec.preprocessing import baseline_als
>>> X_corrected = baseline_als(X)
ValueError: operands could not be broadcast together with shapes (100, 1800) (1801,)
```

**Diagnosis:**

```python
import numpy as np

print(f"X shape: {X.shape}")  # e.g., (100, 1800)
print(f"wavenumbers shape: {wavenumbers.shape}")  # e.g., (1801,)

# Problem: wavenumbers has 1801 elements, but X has 1800 columns
```

**Causes & Solutions:**

#### 1. Wavenumber Grid Mismatch

**Solution:** Ensure wavenumber array matches spectral columns:
```python
# Check alignment
assert X.shape[1] == len(wavenumbers), f"Mismatch: {X.shape[1]} vs {len(wavenumbers)}"

# If mismatch, trim or interpolate
if X.shape[1] != len(wavenumbers):
    # Option A: Trim wavenumbers to match X
    wavenumbers = wavenumbers[:X.shape[1]]
    
    # Option B: Trim X to match wavenumbers
    X = X[:, :len(wavenumbers)]
    
    # Option C: Interpolate to common grid (recommended)
    from foodspec.preprocessing import interpolate_to_grid
    X, wavenumbers = interpolate_to_grid(X, wavenumbers, new_grid=np.arange(4000, 650, -2))
```

#### 2. Row vs. Column Confusion

**Problem:** Transpose needed
```python
# Wrong: X is (n_wavenumbers, n_samples) instead of (n_samples, n_wavenumbers)
print(X.shape)  # (1800, 100) - WRONG

# Solution: Transpose
X = X.T
print(X.shape)  # (100, 1800) - CORRECT
```

**FoodSpec convention:** Rows = samples, Columns = wavenumbers

#### 3. 1D vs. 2D Array

**Problem:** Single spectrum treated as 2D
```python
# Wrong
single_spectrum = X[0]  # Shape: (1800,)
baseline_als(single_spectrum)  # Error: expects 2D

# Solution: Reshape to 2D
single_spectrum = X[0:1]  # Shape: (1, 1800)
# OR
single_spectrum = X[0].reshape(1, -1)
baseline_als(single_spectrum)  # Works
```

---

### Problem: "Axis out of range" errors

**Symptoms:**
```python
>>> from foodspec.ml import fit_pls
>>> model = fit_pls(X, y, n_components=10)
numpy.AxisError: axis 1 is out of bounds for array of dimension 1
```

**Diagnosis:**
```python
print(f"X ndim: {X.ndim}")  # Should be 2
print(f"X shape: {X.shape}")
print(f"y ndim: {y.ndim}")  # Should be 1 for labels
```

**Solution:**
```python
# Ensure X is 2D
if X.ndim == 1:
    X = X.reshape(1, -1)  # Single sample

# Ensure y is 1D (for classification/regression)
if y.ndim > 1:
    y = y.ravel()  # Flatten (100, 1) → (100,)
```

---

<a id="preprocessing-issues"></a>
## NaNs After Preprocessing

### Problem: NaNs appear after baseline correction

**Symptoms:**
```python
>>> X_corrected = baseline_als(X, lam=1e6, p=0.01)
>>> np.isnan(X_corrected).sum()
1500  # Many NaNs!
```

**Causes & Solutions:**

#### 1. Input Contains NaNs

**Check input:**
```python
print(f"NaNs in input: {np.isnan(X).sum()}")
```

**Solution:** Remove or impute NaNs before preprocessing:
```python
# Option A: Drop samples with NaNs
mask = ~np.isnan(X).any(axis=1)
X_clean = X[mask]

# Option B: Impute with interpolation
from scipy.interpolate import interp1d
for i in range(X.shape[0]):
    if np.isnan(X[i]).any():
        nan_mask = np.isnan(X[i])
        not_nan = ~nan_mask
        X[i, nan_mask] = np.interp(
            np.where(nan_mask)[0],
            np.where(not_nan)[0],
            X[i, not_nan]
        )
```

#### 2. Division by Zero in Normalization

**Problem:** Zero or near-zero standard deviation
```python
# SNV normalization: (X - mean) / std
# If std ≈ 0 → division by zero → NaN

from foodspec.preprocessing import snv
X_norm = snv(X)

# Diagnosis
stds = X.std(axis=1, ddof=1)
print(f"Samples with std < 1e-6: {(stds < 1e-6).sum()}")
```

**Solution:** Add epsilon to denominator or filter flat spectra:
```python
# Option A: Filter flat spectra
threshold = 1e-4
mask = X.std(axis=1, ddof=1) > threshold
X_filtered = X[mask]
X_norm = snv(X_filtered)

# Option B: Custom SNV with epsilon
def snv_safe(X, eps=1e-8):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, ddof=1, keepdims=True)
    std = np.where(std < eps, eps, std)  # Avoid division by zero
    return (X - mean) / std

X_norm = snv_safe(X)
```

#### 3. Baseline Correction Failure

**Problem:** ALS baseline correction fails on saturated spectra
```python
# Diagnosis: Check for saturation
print(f"Max intensity: {X.max()}")
print(f"Saturated pixels: {(X > 3.0).sum()}")  # Absorbance > 3.0
```

**Solution:** Clip intensities or skip saturated spectra:
```python
# Clip absorbance to reasonable range
X_clipped = np.clip(X, -0.5, 3.0)
X_corrected = baseline_als(X_clipped, lam=1e6, p=0.01)
```

---

### Problem: NaNs after derivative calculation

**Symptoms:**
```python
>>> from foodspec.preprocessing import savgol_filter
>>> X_deriv = savgol_filter(X, window_length=11, polyorder=2, deriv=1)
>>> np.isnan(X_deriv).sum()
50  # NaNs at edges
```

**Cause:** Edge effects in Savitzky-Golay filter

**Solution:** Use `mode='interp'` or trim edges:
```python
from scipy.signal import savgol_filter

# Option A: Use interp mode (extrapolates to edges)
X_deriv = savgol_filter(X, window_length=11, polyorder=2, deriv=1, mode='interp', axis=1)

# Option B: Trim edges
window = 11
edge = window // 2
X_deriv = savgol_filter(X, window_length=window, polyorder=2, deriv=1, axis=1)
X_deriv = X_deriv[:, edge:-edge]  # Remove edge columns
wavenumbers = wavenumbers[edge:-edge]  # Also trim wavenumbers
```

---

<a id="model-building-issues"></a>
## Model Overfitting / Too-Good Accuracy

### Problem: Suspiciously high accuracy (>95%)

**Symptoms:**
```python
>>> from sklearn.model_selection import cross_val_score
>>> scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
>>> scores.mean()
0.989  # 98.9% accuracy - too good to be true!
```

**Diagnosis Checklist:**

```python
# 1. Check for data leakage (replicate leakage)
print(f"Number of samples: {len(np.unique(sample_ids))}")
print(f"Number of spectra: {len(X)}")
print(f"Replicates per sample: {len(X) / len(np.unique(sample_ids))}")

# 2. Check train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 3. Check class balance
from collections import Counter
print(f"Class distribution: {Counter(y)}")

# 4. Check for preprocessing leakage
# Did you normalize BEFORE splitting? That's leakage!
```

**Common Causes & Solutions:**

#### 1. Replicate Leakage

**Problem:** Technical replicates split across train/test

**Diagnosis:**
```python
from sklearn.model_selection import GroupKFold, KFold

# Random CV (leaky)
cv_random = KFold(n_splits=5, shuffle=True, random_state=42)
scores_random = cross_val_score(model, X, y, cv=cv_random)

# Grouped CV (correct)
cv_grouped = GroupKFold(n_splits=5)
scores_grouped = cross_val_score(model, X, y, cv=cv_grouped, groups=sample_ids)

print(f"Random CV:  {scores_random.mean():.3f}")
print(f"Grouped CV: {scores_grouped.mean():.3f}")
print(f"Drop: {scores_random.mean() - scores_grouped.mean():.3f}")

# If drop > 0.10 → replicate leakage!
```

**Solution:** Always use grouped CV
```python
from foodspec.ml.validation import grouped_cross_validation

results = grouped_cross_validation(
    X, y,
    groups=sample_ids,  # Critical!
    model=RandomForestClassifier(),
    n_splits=5,
    n_repeats=10
)
print(f"Realistic Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_ci']:.3f}")
```

#### 2. Preprocessing Leakage

**Problem:** Normalization fit on entire dataset before splitting

**Wrong:**
```python
# ❌ WRONG: Preprocessing before splitting
X_norm = snv(X)  # Uses statistics from entire dataset
X_train, X_test = train_test_split(X_norm, y)
model.fit(X_train, y_train)
```

**Correct:**
```python
# ✅ CORRECT: Preprocessing within CV folds
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),  # Fit only on train in each fold
    ('model', RandomForestClassifier())
])

scores = cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=sample_ids)
```

#### 3. Overfitting Small Datasets

**Problem:** More features than samples (p >> n)

**Diagnosis:**
```python
n_samples, n_features = X.shape
print(f"Samples: {n_samples}, Features: {n_features}")
print(f"Feature-to-sample ratio: {n_features / n_samples:.1f}")

# If ratio > 10 → high overfitting risk
```

**Solution:** Reduce features or regularize
```python
# Option A: Feature selection (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=min(50, n_samples // 2))
X_reduced = pca.fit_transform(X)

# Option B: Regularized models
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l1', solver='saga', C=0.1)  # L1 regularization

# Option C: Simpler models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()  # Good for p >> n
```

---

## Domain Shift Failure

### Problem: Model fails on new batches/instruments

**Symptoms:**
```python
# Trained on Batch 1-4, tested on Batch 5
>>> model.fit(X_train, y_train)  # Batches 1-4
>>> accuracy_train = model.score(X_train, y_train)
>>> accuracy_test = model.score(X_test, y_test)  # Batch 5
>>> print(f"Train: {accuracy_train:.3f}, Test: {accuracy_test:.3f}")
Train: 0.93, Test: 0.65  # 28% drop!
```

**Diagnosis:**

```python
# Visualize batch separation (PCA)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
for batch in np.unique(batches):
    mask = batches == batch
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Batch {batch}', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('Batch Effect Visualization')
plt.show()

# If batches form distinct clusters → domain shift
```

**Causes & Solutions:**

#### 1. Instrument/Day Variability

**Solution A: Batch Correction (ComBat)**
```python
from neuroCombat import neuroCombat

# Harmonize spectra to remove batch effects
X_harmonized = neuroCombat(
    dat=X.T,  # Features × samples
    covars={'batch': batches},
    categorical_cols=['batch']
)['data'].T

# Re-train on harmonized data
model.fit(X_harmonized_train, y_train)
accuracy_harmonized = model.score(X_harmonized_test, y_test)
print(f"Post-Harmonization: {accuracy_harmonized:.3f}")
```

**Solution B: Transfer Learning**
```python
from foodspec.ml.harmonization import transfer_component_analysis

# Align source (old batches) to target (new batch)
X_aligned = transfer_component_analysis(
    X_source=X_train,
    X_target=X_test,
    n_components=10
)

# Re-train on aligned data
model.fit(X_aligned_train, y_train)
```

**Solution C: Standard Addition (Calibration Transfer)**
```python
# Measure standard samples on both instruments
# Use piecewise direct standardization (PDS)
from foodspec.ml.calibration import piecewise_direct_standardization

X_test_corrected = piecewise_direct_standardization(
    X_source=X_train_standards,
    X_target=X_test_standards,
    X_to_correct=X_test,
    window_size=9
)
```

#### 2. Temperature/Humidity Drift

**Solution:** Include environmental covariates or normalize by reference
```python
# Option A: Reference normalization (MSC to reference spectrum)
from foodspec.preprocessing import msc

reference = X_train.mean(axis=0)  # Use training mean as reference
X_train_norm = msc(X_train, reference=reference)
X_test_norm = msc(X_test, reference=reference)

# Option B: Model environmental variables
import pandas as pd
metadata = pd.DataFrame({
    'temperature': [...],
    'humidity': [...],
    'spectrum': X.tolist()
})

# Include as features or stratify
```

---

## Reproducibility Mismatch

### Problem: Results differ across runs despite setting seed

**Symptoms:**
```python
# Run 1
>>> np.random.seed(42)
>>> scores1 = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5)
>>> scores1.mean()
0.873

# Run 2 (same code)
>>> np.random.seed(42)
>>> scores2 = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5)
>>> scores2.mean()
0.879  # Different!
```

**Causes & Solutions:**

#### 1. Missing Random State in CV

**Problem:** CV splitter not seeded
```python
# Wrong
cv = KFold(n_splits=5, shuffle=True)  # No random_state!

# Correct
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

#### 2. Library Version Mismatch

**Check versions:**
```python
import sklearn, numpy, scipy, foodspec

print(f"scikit-learn: {sklearn.__version__}")
print(f"numpy: {numpy.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"foodspec: {foodspec.__version__}")
```

**Solution:** Document and freeze versions
```bash
# Save environment
pip freeze > requirements.txt

# Or use conda
conda env export > environment.yml

# Share requirements.txt with collaborators
```

**Known version-dependent behaviors:**
- **NumPy <1.20 vs ≥1.20:** RNG changed (use `np.random.Generator` for consistency)
- **scikit-learn 0.24 vs 1.0+:** `random_state` behavior changed in some estimators

#### 3. Parallelism Non-Determinism

**Problem:** `n_jobs=-1` causes non-deterministic behavior

**Solution:** Set `n_jobs=1` for reproducibility (slower but deterministic)
```python
# For reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)

# For speed (may not be perfectly reproducible)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

#### 4. Floating-Point Precision

**Problem:** Different hardware (CPU vs GPU, Intel vs ARM) gives slightly different results

**Solution:** Accept small differences (<1e-6) or use lower precision
```python
# Compare with tolerance
np.testing.assert_allclose(result1, result2, rtol=1e-5, atol=1e-6)

# Round for comparison
result1_rounded = np.round(result1, decimals=5)
result2_rounded = np.round(result2, decimals=5)
assert np.allclose(result1_rounded, result2_rounded)
```

---

## Quick Diagnostic Script

Run this script to diagnose common issues:

```python
import numpy as np
import sys

def diagnose_data(X, y=None, wavenumbers=None, sample_ids=None):
    """Comprehensive data diagnostics."""
    
    print("="*60)
    print("FOODSPEC DATA DIAGNOSTICS")
    print("="*60)
    
    # 1. Shape checks
    print(f"\n[1] SHAPE CHECKS")
    print(f"   X shape: {X.shape}")
    print(f"   X dtype: {X.dtype}")
    if y is not None:
        print(f"   y shape: {y.shape}")
        print(f"   y dtype: {y.dtype}")
    if wavenumbers is not None:
        print(f"   wavenumbers shape: {wavenumbers.shape}")
        if X.shape[1] != len(wavenumbers):
            print(f"   ⚠️  WARNING: Shape mismatch! {X.shape[1]} != {len(wavenumbers)}")
    
    # 2. Missing values
    print(f"\n[2] MISSING VALUES")
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    print(f"   NaNs: {n_nan} ({100*n_nan/X.size:.2f}%)")
    print(f"   Infs: {n_inf} ({100*n_inf/X.size:.2f}%)")
    if n_nan > 0 or n_inf > 0:
        print(f"   ⚠️  WARNING: Missing/infinite values detected!")
    
    # 3. Intensity range
    print(f"\n[3] INTENSITY RANGE")
    print(f"   Min: {X.min():.4f}")
    print(f"   Max: {X.max():.4f}")
    print(f"   Mean: {X.mean():.4f}")
    print(f"   Std: {X.std():.4f}")
    if X.max() > 5.0:
        print(f"   ⚠️  WARNING: Unusually high absorbance (>5.0)")
    if X.min() < -1.0:
        print(f"   ⚠️  WARNING: Negative absorbance (<-1.0)")
    
    # 4. Flat spectra
    print(f"\n[4] FLAT SPECTRA CHECK")
    stds = X.std(axis=1, ddof=1)
    n_flat = (stds < 1e-4).sum()
    print(f"   Flat spectra (std < 1e-4): {n_flat} ({100*n_flat/len(X):.2f}%)")
    if n_flat > 0:
        print(f"   ⚠️  WARNING: Flat spectra may cause normalization issues")
    
    # 5. Class balance (if labels provided)
    if y is not None:
        print(f"\n[5] CLASS BALANCE")
        from collections import Counter
        counts = Counter(y)
        for label, count in sorted(counts.items()):
            print(f"   {label}: {count} ({100*count/len(y):.1f}%)")
        min_count = min(counts.values())
        max_count = max(counts.values())
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 3:
            print(f"   ⚠️  WARNING: Severe class imbalance (ratio: {imbalance_ratio:.1f})")
    
    # 6. Replicate structure (if sample_ids provided)
    if sample_ids is not None:
        print(f"\n[6] REPLICATE STRUCTURE")
        n_samples = len(np.unique(sample_ids))
        n_spectra = len(sample_ids)
        avg_reps = n_spectra / n_samples
        print(f"   Unique samples: {n_samples}")
        print(f"   Total spectra: {n_spectra}")
        print(f"   Avg replicates/sample: {avg_reps:.1f}")
        if avg_reps > 1:
            print(f"   ⚠️  IMPORTANT: Use grouped CV to prevent replicate leakage!")
    
    # 7. Feature-to-sample ratio
    print(f"\n[7] OVERFITTING RISK")
    n_samples, n_features = X.shape
    ratio = n_features / n_samples
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Feature/Sample ratio: {ratio:.1f}")
    if ratio > 10:
        print(f"   ⚠️  WARNING: High overfitting risk (p >> n). Consider PCA/feature selection.")
    
    # 8. System info
    print(f"\n[8] SYSTEM INFO")
    print(f"   Python: {sys.version.split()[0]}")
    try:
        import sklearn, scipy, foodspec
        print(f"   NumPy: {np.__version__}")
        print(f"   SciPy: {scipy.__version__}")
        print(f"   scikit-learn: {sklearn.__version__}")
        print(f"   FoodSpec: {foodspec.__version__}")
    except ImportError as e:
        print(f"   ⚠️  Missing package: {e}")
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)

# Usage:
# diagnose_data(X, y, wavenumbers, sample_ids)
```

---

## Still Having Issues?

If your problem isn't covered here:

1. **Check the FAQ:** [Common questions answered](faq.md)
2. **Search existing issues:** [GitHub Issues](https://github.com/chandrasekarnarayana/foodspec/issues)
3. **Ask for help:** [Open a new issue](https://github.com/chandrasekarnarayana/foodspec/issues/new) with:
   - FoodSpec version (`import foodspec; print(foodspec.__version__)`)
   - Python version (`python --version`)
   - Minimal reproducible example
   - Full error traceback
4. **Consult documentation:**
   - [User Guide](../user-guide/index.md)
   - [API Reference](../api/index.md)
   - [Validation & Scientific Rigor](../methods/validation/index.md)

---

## Related Pages

- [FAQ](faq.md) – Frequently asked questions
- [Validation → Leakage Prevention](../methods/validation/cross_validation_and_leakage.md) – Prevent data leakage
- [Reference → Data Format](../reference/data_format.md) – Data validation checklist
- [Troubleshooting FAQ](../troubleshooting/troubleshooting_faq.md) – Additional common issues
- [Reporting Guidelines](../troubleshooting/reporting_guidelines.md) – How to report bugs
