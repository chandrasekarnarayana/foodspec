# Chemometrics API Reference

!!! info "Module Purpose"
    Chemometric models (PCA, PLS-DA, PLS-R), mixture analysis (MCR-ALS), VIP scores, and model validation utilities.

---

## Quick Navigation

| Function/Class | Purpose | Typical Use |
|----------------|---------|-------------|
| [`run_pca()`](#pca-analysis) | Principal Component Analysis | Dimensionality reduction, outlier detection |
| [`make_pls_da()`](#classification-models) | PLS-DA classifier | Classification with interpretable loadings |
| [`make_pls_regression()`](#regression-models) | PLS regression | Quantitative analysis |
| [`mcr_als()`](#mixture-analysis) | Multivariate Curve Resolution | Pure component extraction |
| [`calculate_vip()`](#variable-importance) | VIP scores | Feature importance for PLS models |
| [`cross_validate_pipeline()`](#validation) | Cross-validation | Model evaluation |

---

## Common Patterns

### Pattern 1: PCA for Outlier Detection

```python
from foodspec.chemometrics import run_pca
from foodspec.io import load_folder

# Load data
fs = load_folder('data/olive_oils/')

# Run PCA
pca_result = run_pca(fs, n_components=5)

# Analyze results
print(f"Explained variance: {pca_result.explained_variance_ratio_[:3]}")
print(f"PC1 explains {pca_result.explained_variance_ratio_[0]*100:.1f}%")

# Detect outliers using Hotelling's T²
scores = pca_result.transform(fs.x)
t2 = (scores**2).sum(axis=1)
outliers = fs[t2 > t2.mean() + 3*t2.std()]
print(f"Found {len(outliers)} outliers")
```

### Pattern 2: PLS-DA Classification

```python
from foodspec.chemometrics import make_pls_da
from foodspec.ml import nested_cross_validate

# Create PLS-DA classifier
clf = make_pls_da(n_components=10)

# Cross-validate
cv_results = nested_cross_validate(
    fs, clf,
    target_col='variety',
    outer_cv=5,
    inner_cv=3
)

print(f"Accuracy: {cv_results['test_accuracy'].mean():.3f}")
print(f"F1-score: {cv_results['test_f1_macro'].mean():.3f}")
```

### Pattern 3: Mixture Analysis (MCR-ALS)

```python
from foodspec.chemometrics import mcr_als

# Pure component extraction
result = mcr_als(
    D=fs.x,  # Mixture spectra
    n_components=3,
    max_iter=100,
    tol=1e-6
)

# Get pure components and concentrations
C = result['concentrations']  # (n_samples, n_components)
ST = result['spectra']  # (n_components, n_wavenumbers)

print(f"Residual: {result['residual']:.4f}")
print(f"Iterations: {result['n_iter']}")
```

### Pattern 4: VIP-Based Feature Selection

```python
from foodspec.chemometrics import make_pls_regression, calculate_vip

# Train PLS regression
pls = make_pls_regression(n_components=10)
pls.fit(fs.x, fs.metadata['concentration'])

# Calculate VIP scores
vip = calculate_vip(pls, fs.x, fs.metadata['concentration'])

# Select important features (VIP > 1)
important_wn = fs.wavenumbers[vip > 1.0]
print(f"Selected {len(important_wn)} important wavenumbers")

# Use for feature selection
fs_reduced = fs.select_wavenumbers(important_wn)
```

---

## PCA Analysis

### run_pca

Principal Component Analysis for dimensionality reduction and exploratory analysis.

::: foodspec.chemometrics.run_pca
    options:
      show_source: false
      heading_level: 4

**Example:**

```python
from foodspec.chemometrics import run_pca

# Run PCA
pca = run_pca(fs, n_components=10)

# Access results
scores = pca.transform(fs.x)
loadings = pca.components_
explained_var = pca.explained_variance_ratio_
```

---

## Classification Models

### make_pls_da

Create a PLS-DA (Partial Least Squares Discriminant Analysis) classifier.

::: foodspec.chemometrics.make_pls_da
    options:
      show_source: false
      heading_level: 4

### make_simca

Create a SIMCA (Soft Independent Modeling of Class Analogy) classifier.

::: foodspec.chemometrics.make_simca
    options:
      show_source: false
      heading_level: 4

---

## Regression Models

### make_pls_regression

Create a PLS (Partial Least Squares) regression model.

::: foodspec.chemometrics.make_pls_regression
    options:
      show_source: false
      heading_level: 4

---

## Mixture Analysis

### mcr_als

Multivariate Curve Resolution using Alternating Least Squares.

::: foodspec.chemometrics.mcr_als
    options:
      show_source: false
      heading_level: 4

### run_mixture_analysis_workflow

High-level workflow for mixture analysis with visualization.

::: foodspec.chemometrics.run_mixture_analysis_workflow
    options:
      show_source: false
      heading_level: 4

---

## Variable Importance

### calculate_vip

Calculate Variable Importance in Projection (VIP) scores for PLS models.

::: foodspec.chemometrics.calculate_vip
    options:
      show_source: false
      heading_level: 4

### calculate_vip_da

VIP scores for PLS-DA classification models.

::: foodspec.chemometrics.calculate_vip_da
    options:
      show_source: false
      heading_level: 4

---

## Validation

### cross_validate_pipeline

Cross-validation for preprocessing + model pipelines.

::: foodspec.chemometrics.cross_validate_pipeline
    options:
      show_source: false
      heading_level: 4

### compute_explained_variance

Compute explained variance for dimensionality reduction models.

::: foodspec.chemometrics.compute_explained_variance
    options:
      show_source: false
      heading_level: 4

---

## Cross-References

**Related Modules:**
- [Core](core.md) - FoodSpectrumSet data structure
- [Preprocessing](preprocessing.md) - Preprocess before modeling
- [ML](ml.md) - Model training and evaluation

**Related Workflows:**
- [Oil Authentication](../workflows/authentication/oil_authentication.md) - PLS-DA for classification
- [Mixture Analysis](../workflows/quantification/mixture_analysis.md) - MCR-ALS workflow
