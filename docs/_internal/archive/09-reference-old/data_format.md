# Data Format Reference

!!! info "Module Purpose"
    Comprehensive guide to FoodSpec data structures, format conventions, unit standards, and validation checklists.

---

## Overview

FoodSpec accepts spectral data in multiple formats. This guide defines:
- **Schema formats** (wide vs. long/tidy)
- **Unit conventions** and allowed ranges
- **Metadata requirements**
- **Data validation checklist** for pre-modeling QC

---

## Schema Formats

### Format 1: Wide Format (Recommended for FoodSpec)

**Structure:** Each row = one spectrum, columns = metadata + wavenumber values

**Schema:**

| sample_id | batch | variety | replicate | wn_4000 | wn_3998 | ... | wn_650 |
|-----------|-------|---------|-----------|---------|---------|-----|--------|
| OO_001    | B1    | EVOO    | 1         | 0.234   | 0.235   | ... | 0.012  |
| OO_001    | B1    | EVOO    | 2         | 0.236   | 0.237   | ... | 0.013  |
| OO_002    | B1    | EVOO    | 1         | 0.241   | 0.242   | ... | 0.011  |

**Advantages:**
- ✅ Compact (one row per spectrum)
- ✅ Direct input to scikit-learn, FoodSpec APIs
- ✅ Efficient memory usage
- ✅ Easy to visualize with standard plotting tools

**Requirements:**
- Column names for wavenumbers must be numeric or parseable (e.g., `wn_4000`, `4000`, `wn4000`)
- All spectra must have same wavenumber grid
- Missing values: Use `NaN` (avoid zeros which are real measurements)

**Example (CSV):**

```csv
sample_id,batch,variety,replicate,4000.0,3998.0,3996.0,...,650.0
OO_001,B1,EVOO,1,0.234,0.235,0.236,...,0.012
OO_001,B1,EVOO,2,0.236,0.237,0.238,...,0.013
```

**Load with FoodSpec:**

```python
from foodspec.io import load_from_metadata_table
import pandas as pd

# Load CSV
df = pd.read_csv('spectra_wide.csv')

# Separate metadata and spectra
metadata_cols = ['sample_id', 'batch', 'variety', 'replicate']
wn_cols = [col for col in df.columns if col not in metadata_cols]

metadata = df[metadata_cols]
X = df[wn_cols].values
wavenumbers = np.array([float(col) for col in wn_cols])

# Create FoodSpectrumSet
fs = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=metadata)
```

---

### Format 2: Long/Tidy Format

**Structure:** Each row = one (wavenumber, intensity) pair

**Schema:**

| sample_id | batch | variety | replicate | wavenumber | intensity |
|-----------|-------|---------|-----------|------------|-----------|
| OO_001    | B1    | EVOO    | 1         | 4000.0     | 0.234     |
| OO_001    | B1    | EVOO    | 1         | 3998.0     | 0.235     |
| OO_001    | B1    | EVOO    | 1         | 3996.0     | 0.236     |
| ...       | ...   | ...     | ...       | ...        | ...       |
| OO_001    | B1    | EVOO    | 1         | 650.0      | 0.012     |
| OO_001    | B1    | EVOO    | 2         | 4000.0     | 0.236     |

**Advantages:**
- ✅ Database-friendly (SQL, data warehouses)
- ✅ Flexible (irregular wavenumber grids per sample)
- ✅ Easy filtering by wavenumber range
- ✅ Compatible with tidyverse (R), pandas melt/pivot

**Disadvantages:**
- ❌ Large file size (~1000x rows for 1000 wavenumber points)
- ❌ Requires pivoting before modeling
- ❌ Slower I/O

**Example (CSV):**

```csv
sample_id,batch,variety,replicate,wavenumber,intensity
OO_001,B1,EVOO,1,4000.0,0.234
OO_001,B1,EVOO,1,3998.0,0.235
OO_001,B1,EVOO,1,3996.0,0.236
```

**Convert to Wide Format:**

```python
import pandas as pd

# Load long format
df_long = pd.read_csv('spectra_long.csv')

# Pivot to wide format
df_wide = df_long.pivot_table(
    index=['sample_id', 'batch', 'variety', 'replicate'],
    columns='wavenumber',
    values='intensity'
).reset_index()

# Now load as wide format
```

**Export from FoodSpec to Long Format:**

```python
from foodspec.io import to_tidy_csv

# Export FoodSpectrumSet to tidy/long CSV
to_tidy_csv(fs, 'spectra_long.csv')
```

---

## Unit Conventions

### Wavenumber

**Unit:** cm⁻¹ (reciprocal centimeters)

**Allowed ranges:**
- **FTIR-ATR:** 4000 - 650 cm⁻¹ (typical), 4000 - 400 cm⁻¹ (extended)
- **FTIR-Transmission:** 4000 - 400 cm⁻¹
- **Raman:** Depends on laser (e.g., 532 nm laser: 200 - 4000 cm⁻¹ shift)
- **NIR:** 12500 - 4000 cm⁻¹ (800 - 2500 nm wavelength)

**Validation:**
- Must be strictly ascending: `wavenumbers[i] < wavenumbers[i+1]`
- Must be numeric (no NaN, inf)
- Resolution typically 2-8 cm⁻¹ for FTIR, 1-4 cm⁻¹ for Raman

**Conversion from wavelength (nm):**

$$\tilde{\nu}\ (\text{cm}^{-1}) = \frac{10^7}{\lambda\ (\text{nm})}$$

Example: 2500 nm → 4000 cm⁻¹

---

### Intensity

**Units:** Arbitrary (absorbance, reflectance, counts, a.u.)

**Allowed ranges:**
- **Absorbance:** -0.5 to 5.0 (typical), -2.0 to 10.0 (extreme)
  - Negative values possible due to baseline/noise
  - >3.0 indicates saturation (non-linear regime)
- **Reflectance:** 0.0 to 1.0 (fractional) or 0 to 100 (percentage)
- **Transmittance:** 0.0 to 1.0
- **Counts (Raman):** 0 to instrument max (e.g., 65535 for 16-bit)

**Preprocessing recommendations:**
- If reflectance, convert to absorbance: $A = -\log_{10}(R)$
- If transmittance, convert to absorbance: $A = -\log_{10}(T)$
- Clip extreme values before modeling (e.g., abs(intensity) > 10 likely errors)

**Normalization types:**
- **Raw:** Instrument output (use for troubleshooting only)
- **SNV-normalized:** Mean=0, std=1 per spectrum (removes multiplicative scatter)
- **Reference-peak normalized:** Divided by intensity at reference wavenumber
- **Vector-normalized:** Divided by L2 norm ($||I||_2 = 1$)

---

### Metadata Fields

**Required fields:**
- `sample_id` (str): Unique identifier for each independent sample
  - Example: `OO_001`, `EVOO_batch3_001`
  - Use hierarchical IDs: `{type}_{batch}_{number}`

**Recommended fields:**
- `batch` (str): Instrument batch, measurement date, or operator
  - Critical for batch-aware CV and harmonization
- `replicate` (int): Technical replicate number (1, 2, 3, ...)
  - Keep replicates together in CV folds (use `sample_id` as group)
- `label` / `variety` / `class` (str): Ground truth label for supervised learning
- `timestamp` (ISO 8601): Measurement date/time for temporal analysis
- `instrument_id` (str): Instrument serial number for multi-instrument studies

**Optional fields:**
- `concentration`, `temperature`, `time`, `pH`, `moisture_content` (numeric)
- `treatment`, `origin`, `species` (categorical)

**Validation:**
- No duplicate `(sample_id, replicate)` pairs
- `batch` must be consistent within `sample_id` (replicates measured together)
- `label` encoding: Use strings (not integers) to avoid confusion
  - Good: `"EVOO"`, `"Lampante"`, `"Adulterated"`
  - Bad: `1`, `2`, `3` (ambiguous ordering)

---

## Data Validation Checklist

Run this checklist **before modeling** to catch data quality issues early:

### ✅ Step 1: Load and Inspect

```python
import pandas as pd
from foodspec.io import load_from_metadata_table

# Load data
df = pd.read_csv('spectra.csv')

# Basic inspection
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Dtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
```

**Check:**
- ✅ Expected number of rows (samples × replicates)
- ✅ Expected wavenumber range (columns)
- ✅ No completely empty rows/columns
- ✅ Metadata columns are correct dtypes (str for categorical, float for numeric)

---

### ✅ Step 2: Wavenumber Grid Validation

```python
import numpy as np

# Extract wavenumber columns
metadata_cols = ['sample_id', 'batch', 'variety', 'replicate']
wn_cols = [col for col in df.columns if col not in metadata_cols]
wavenumbers = np.array([float(col) for col in wn_cols])

# Check 1: Ascending order
assert np.all(np.diff(wavenumbers) > 0), "Wavenumbers not strictly ascending!"

# Check 2: Reasonable range
wn_min, wn_max = wavenumbers.min(), wavenumbers.max()
print(f"Wavenumber range: {wn_min:.1f} - {wn_max:.1f} cm⁻¹")
assert 200 <= wn_min <= 5000, f"Unusual min wavenumber: {wn_min}"
assert 500 <= wn_max <= 15000, f"Unusual max wavenumber: {wn_max}"

# Check 3: Resolution
resolution = np.median(np.diff(wavenumbers))
print(f"Median resolution: {resolution:.2f} cm⁻¹")
assert 0.5 <= resolution <= 20, f"Unusual resolution: {resolution}"
```

**Expected output:**
- FTIR: 4000-650 cm⁻¹, resolution ~4 cm⁻¹
- Raman: 200-4000 cm⁻¹, resolution ~2 cm⁻¹
- NIR: 4000-12500 cm⁻¹, resolution ~8 cm⁻¹

---

### ✅ Step 3: Intensity Validation

```python
# Extract intensity matrix
X = df[wn_cols].values

# Check 1: No NaN/inf
n_nan = np.isnan(X).sum()
n_inf = np.isinf(X).sum()
print(f"NaN values: {n_nan}, Inf values: {n_inf}")
assert n_nan == 0 and n_inf == 0, "Found NaN or Inf values!"

# Check 2: Reasonable range
intensity_min, intensity_max = X.min(), X.max()
print(f"Intensity range: {intensity_min:.3f} to {intensity_max:.3f}")

# Warning for extreme values
if intensity_min < -2.0:
    print(f"⚠️  Warning: Very negative intensities ({intensity_min:.3f})")
if intensity_max > 5.0:
    print(f"⚠️  Warning: Very high intensities ({intensity_max:.3f}) - possible saturation")

# Check 3: Non-constant spectra
spectrum_std = X.std(axis=1)
print(f"Spectrum std dev: min={spectrum_std.min():.4f}, max={spectrum_std.max():.4f}")
assert (spectrum_std > 0).all(), "Found constant spectra (all values identical)!"
```

**Red flags:**
- ❌ Intensity all zeros or all ones → measurement failure
- ❌ Std dev < 0.001 → flat spectra, no information
- ❌ Absorbance > 3.0 → saturation (non-linear regime)

---

### ✅ Step 4: Metadata Validation

```python
# Check 1: Required columns exist
required_cols = ['sample_id', 'batch', 'replicate']
for col in required_cols:
    assert col in df.columns, f"Missing required column: {col}"

# Check 2: No duplicate (sample_id, replicate) pairs
duplicates = df.duplicated(subset=['sample_id', 'replicate'])
print(f"Duplicate (sample_id, replicate): {duplicates.sum()}")
assert not duplicates.any(), "Found duplicate sample/replicate pairs!"

# Check 3: Consistent batch within sample_id
batch_consistency = df.groupby('sample_id')['batch'].nunique()
inconsistent = batch_consistency[batch_consistency > 1]
if len(inconsistent) > 0:
    print(f"⚠️  Warning: {len(inconsistent)} samples have multiple batches")
    print(inconsistent)

# Check 4: Class balance (if classification)
if 'variety' in df.columns:
    class_counts = df['variety'].value_counts()
    print(f"\nClass distribution:\n{class_counts}")
    
    # Warn if imbalanced
    min_class = class_counts.min()
    max_class = class_counts.max()
    if max_class / min_class > 3:
        print(f"⚠️  Warning: Imbalanced classes (ratio {max_class/min_class:.1f}:1)")
        print("   Consider stratified CV or resampling")
```

**Red flags:**
- ❌ Duplicate samples → data entry error or improper merging
- ❌ Class ratio > 10:1 → use balanced accuracy, stratified CV, or SMOTE
- ❌ Only 1 replicate per sample → can't estimate technical variance

---

### ✅ Step 5: Replicate Consistency

```python
# Group by sample_id and check replicate variation
replicate_cv = []
for sample_id, group in df.groupby('sample_id'):
    if len(group) > 1:  # Only if replicates exist
        X_replicates = group[wn_cols].values
        mean_spectrum = X_replicates.mean(axis=0)
        cv = (X_replicates.std(axis=0) / (mean_spectrum + 1e-10)).mean()
        replicate_cv.append({'sample_id': sample_id, 'cv': cv})

if replicate_cv:
    cv_df = pd.DataFrame(replicate_cv)
    print(f"\nReplicate CV: median={cv_df['cv'].median():.4f}, max={cv_df['cv'].max():.4f}")
    
    # Flag problematic samples
    high_cv = cv_df[cv_df['cv'] > 0.1]
    if len(high_cv) > 0:
        print(f"⚠️  Warning: {len(high_cv)} samples with high replicate variation (CV > 10%)")
        print(high_cv)
```

**Expected values:**
- Good: Replicate CV < 5% (0.05)
- Acceptable: CV < 10% (0.10)
- Poor: CV > 10% → check sample prep or instrument stability

---

### ✅ Step 6: Outlier Detection

```python
from foodspec.chemometrics import run_pca

# Run PCA
fs = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=df[metadata_cols])
pca = run_pca(fs, n_components=5)

# Compute Hotelling's T²
scores = pca.transform(X)
t2 = (scores**2 / pca.explained_variance_).sum(axis=1)

# Flag outliers (T² > 99th percentile)
threshold = np.percentile(t2, 99)
outliers = t2 > threshold
print(f"\nOutliers detected: {outliers.sum()} / {len(outliers)}")

if outliers.any():
    print("\nOutlier sample IDs:")
    print(df.loc[outliers, 'sample_id'].tolist())
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(scores[:, 0], scores[:, 1], c=outliers, cmap='coolwarm', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Scores (outliers in red)')
    
    plt.subplot(1, 2, 2)
    plt.hist(t2, bins=50)
    plt.axvline(threshold, color='r', linestyle='--', label=f'99th percentile: {threshold:.1f}')
    plt.xlabel("Hotelling's T²")
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

**Action for outliers:**
- Inspect visually: Plot spectrum vs. mean spectrum
- Check metadata: Mislabeled? Different sample type?
- Decision: Remove (if error) or keep (if real variation)

---

### ✅ Step 7: Batch Effect Check

```python
# Visualize batch separation in PCA
if 'batch' in df.columns and df['batch'].nunique() > 1:
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    batch_encoded = le.fit_transform(df['batch'])
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(scores[:, 0], scores[:, 1], c=batch_encoded, cmap='tab10', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA colored by Batch')
    plt.colorbar(scatter, label='Batch')
    
    # ANOVA test for batch effect
    from scipy.stats import f_oneway
    pc1_by_batch = [scores[batch_encoded == i, 0] for i in range(len(le.classes_))]
    f_stat, p_val = f_oneway(*pc1_by_batch)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(pc1_by_batch, labels=le.classes_)
    plt.xlabel('Batch')
    plt.ylabel('PC1 Score')
    plt.title(f'Batch Effect Test (p={p_val:.4f})')
    plt.tight_layout()
    plt.show()
    
    if p_val < 0.05:
        print(f"⚠️  Significant batch effect detected (p={p_val:.4f})")
        print("   Recommendations:")
        print("   - Use batch-aware CV (GroupKFold with batch as group)")
        print("   - Apply harmonization (e.g., multiplicative scatter correction)")
        print("   - Include batch as covariate in model")
```

---

## Complete Validation Script

```python
"""
FoodSpec Data Validation Script
Run this before any modeling to ensure data quality
"""

import numpy as np
import pandas as pd
from foodspec.io import load_from_metadata_table
from foodspec.core import FoodSpectrumSet
from foodspec.chemometrics import run_pca
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def validate_data(csv_path, metadata_cols=['sample_id', 'batch', 'variety', 'replicate']):
    """
    Comprehensive data validation for FoodSpec.
    
    Parameters
    ----------
    csv_path : str
        Path to wide-format CSV file
    metadata_cols : list
        Column names for metadata (non-spectral)
    
    Returns
    -------
    dict
        Validation report with pass/fail status
    """
    print("="*80)
    print("FOODSPEC DATA VALIDATION")
    print("="*80)
    
    report = {'passed': True, 'warnings': [], 'errors': []}
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded data: {df.shape[0]} samples × {df.shape[1]} columns")
    
    # Extract wavenumbers and intensity matrix
    wn_cols = [col for col in df.columns if col not in metadata_cols]
    wavenumbers = np.array([float(col) for col in wn_cols])
    X = df[wn_cols].values
    
    # 1. Wavenumber validation
    print("\n[1/7] Wavenumber Grid Validation")
    if not np.all(np.diff(wavenumbers) > 0):
        report['errors'].append("Wavenumbers not strictly ascending")
        report['passed'] = False
    else:
        print(f"  ✓ Range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
        print(f"  ✓ Resolution: {np.median(np.diff(wavenumbers)):.2f} cm⁻¹")
    
    # 2. Intensity validation
    print("\n[2/7] Intensity Validation")
    if np.isnan(X).any() or np.isinf(X).any():
        report['errors'].append("Found NaN or Inf values")
        report['passed'] = False
    else:
        print(f"  ✓ Range: {X.min():.3f} to {X.max():.3f}")
        if X.max() > 5.0:
            report['warnings'].append(f"High intensity ({X.max():.1f}) - possible saturation")
    
    # 3. Metadata validation
    print("\n[3/7] Metadata Validation")
    for col in ['sample_id', 'batch', 'replicate']:
        if col not in df.columns:
            report['errors'].append(f"Missing required column: {col}")
            report['passed'] = False
    
    duplicates = df.duplicated(subset=['sample_id', 'replicate']).sum()
    if duplicates > 0:
        report['errors'].append(f"Found {duplicates} duplicate (sample_id, replicate) pairs")
        report['passed'] = False
    else:
        print(f"  ✓ No duplicates")
    
    # 4. Class balance
    print("\n[4/7] Class Distribution")
    if 'variety' in df.columns:
        counts = df['variety'].value_counts()
        print(counts)
        if counts.max() / counts.min() > 3:
            report['warnings'].append(f"Imbalanced classes (ratio {counts.max()/counts.min():.1f}:1)")
    
    # 5. Replicate consistency
    print("\n[5/7] Replicate Consistency")
    cv_list = []
    for sid, group in df.groupby('sample_id'):
        if len(group) > 1:
            X_rep = group[wn_cols].values
            cv = (X_rep.std(axis=0) / (X_rep.mean(axis=0) + 1e-10)).mean()
            cv_list.append(cv)
    
    if cv_list:
        median_cv = np.median(cv_list)
        print(f"  Median replicate CV: {median_cv:.4f}")
        if median_cv > 0.1:
            report['warnings'].append(f"High replicate variation (CV={median_cv:.2%})")
    
    # 6. Outlier detection
    print("\n[6/7] Outlier Detection")
    fs = FoodSpectrumSet(x=X, wavenumbers=wavenumbers, metadata=df[metadata_cols])
    pca = run_pca(fs, n_components=5)
    scores = pca.transform(X)
    t2 = (scores**2 / pca.explained_variance_).sum(axis=1)
    outliers = t2 > np.percentile(t2, 99)
    print(f"  Found {outliers.sum()} outliers (>99th percentile T²)")
    if outliers.sum() > len(X) * 0.05:
        report['warnings'].append(f"{outliers.sum()} outliers (>{5}%)")
    
    # 7. Batch effect
    print("\n[7/7] Batch Effect Check")
    if 'batch' in df.columns and df['batch'].nunique() > 1:
        le = LabelEncoder()
        batch_encoded = le.fit_transform(df['batch'])
        pc1_by_batch = [scores[batch_encoded == i, 0] for i in range(len(le.classes_))]
        f_stat, p_val = f_oneway(*pc1_by_batch)
        print(f"  ANOVA p-value: {p_val:.4f}")
        if p_val < 0.05:
            report['warnings'].append(f"Significant batch effect (p={p_val:.4f})")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    if report['passed'] and len(report['warnings']) == 0:
        print("✅ ALL CHECKS PASSED - Data ready for modeling")
    elif report['passed']:
        print(f"⚠️  PASSED WITH {len(report['warnings'])} WARNINGS")
        for w in report['warnings']:
            print(f"   - {w}")
    else:
        print(f"❌ FAILED WITH {len(report['errors'])} ERRORS")
        for e in report['errors']:
            print(f"   - {e}")
    print("="*80)
    
    return report

# Usage
if __name__ == "__main__":
    report = validate_data('your_spectra.csv')
```

---

## Best Practices Summary

### Data Collection
- ✅ **Measure replicates** (≥3 per sample) to quantify technical variation
- ✅ **Record batch/instrument** in metadata for batch-aware CV
- ✅ **Use consistent sample prep** across all samples
- ✅ **Randomize measurement order** to avoid temporal confounding

### Data Format
- ✅ **Use wide format** for FoodSpec (one row per spectrum)
- ✅ **Include sample_id, batch, replicate** in metadata
- ✅ **Keep wavenumbers ascending** and consistent across samples
- ✅ **Avoid integer label encoding** (use descriptive strings)

### Pre-Modeling QC
- ✅ **Run validation script** before any analysis
- ✅ **Check for outliers** (PCA + Hotelling's T²)
- ✅ **Test for batch effects** (ANOVA on PC1)
- ✅ **Verify replicate consistency** (CV < 10%)

### Cross-Validation
- ✅ **Use Group K-fold** with sample_id as group (keeps replicates together)
- ✅ **Perform preprocessing within CV folds** (prevent leakage)
- ✅ **Use batch-aware CV** if multi-instrument study
- ✅ **Report both CV and test set metrics** (nested CV)

---

## Cross-References

**Related Documentation:**
- [Glossary](glossary.md) - Definitions of key terms
- [IO API](../api/io.md) - Loading and saving functions
- [Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md) - Data cleaning
- [Validation Strategies](../methods/validation/advanced_validation_strategies.md) - CV best practices

**Quick Start Guides:**
- [Python API Quickstart](../getting-started/quickstart_python.md)
- [CLI Quickstart](../getting-started/quickstart_cli.md)
- [15-Minute Quickstart](../getting-started/quickstart_15min.md)
