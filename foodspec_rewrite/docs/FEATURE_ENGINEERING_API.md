# Feature Engineering API Reference

## Overview

The FoodSpec feature engineering module provides a clean, sklearn-compatible API for extracting features from spectroscopy data.

## Core Concepts

### FeatureExtractor ABC

All feature extractors inherit from the `FeatureExtractor` abstract base class:

```python
from foodspec.features import FeatureExtractor

class MyExtractor(FeatureExtractor):
    def fit(self, X, y=None, x_grid=None, meta=None):
        """Fit the extractor to training data."""
        # Learn parameters from X
        return self
    
    def transform(self, X, x_grid=None, meta=None):
        """Transform data to feature space."""
        # Extract features from X
        Xf = ...  # shape (n_samples, n_features)
        feature_names = ["f1", "f2", ...]
        return Xf, feature_names
```

**Key Properties:**

- Abstract base class (cannot instantiate directly)
- sklearn-compatible: provides `get_params()` and `set_params()` for GridSearchCV
- `transform()` returns tuple: `(Xf, feature_names)`
- `fit_transform()` provided automatically
- Supports spectroscopy-specific parameters: `x_grid` (wavelengths), `meta` (metadata)

### FeatureSet Container

The `FeatureSet` dataclass wraps extracted features with metadata:

```python
from foodspec.features import FeatureSet

# Create a FeatureSet
feature_set = FeatureSet(
    Xf=np.array([[1, 2], [3, 4]]),          # shape (n_samples, n_features)
    feature_names=["pca_0", "pca_1"],       # length must match n_features
    feature_meta={"extractor": "PCA"}       # optional metadata
)

# Access properties
print(feature_set.n_samples)    # 2
print(feature_set.n_features)   # 2
print(feature_set.Xf.shape)     # (2, 2)

# Select features by index
subset = feature_set.select_features([0])  # Keep only first feature

# Concatenate feature sets
combined = feature_set.concatenate(other_feature_set)
```

**Validation:**

- `Xf` must be 2D numpy array
- `len(feature_names)` must equal `Xf.shape[1]`
- Enforced at creation time via `__post_init__`

## Built-in Extractors

### Peak-Based Features

```python
from foodspec.features import PeakHeights, PeakAreas, PeakRatios

# Extract peak heights at specific wavelengths
extractor = PeakHeights(peak_positions=[1450, 1650])
df = extractor.compute(X, x_grid)  # Returns DataFrame for peaks

# Peak areas via integration
extractor = PeakAreas(peak_positions=[1450], width=20)
df = extractor.compute(X, x_grid)

# Peak ratios for normalization
extractor = PeakRatios(peak_pairs=[(1450, 1650)])
df = extractor.compute(X, x_grid)
```

**Note:** Peak extractors use `compute(X, x_grid)` instead of `fit/transform` since they're stateless.

### Band Integration

```python
from foodspec.features import BandIntegration

# Integrate spectral regions
extractor = BandIntegration(
    bands=[(1400, 1500, "amide_I"), (2800, 3000, "CH_stretch")],
    method="trapz",              # or "sum"
    baseline_subtract=True
)

df = extractor.compute(X, x_grid)
```

### Chemometric Features

```python
from foodspec.features import PCAFeatureExtractor, PLSFeatureExtractor

# PCA dimensionality reduction
pca = PCAFeatureExtractor(n_components=3, random_state=42)
pca.fit(X_train)
Xf, names = pca.transform(X_test)  # Returns (np.ndarray, list)

# PLS (requires labels)
pls = PLSFeatureExtractor(n_components=2, random_state=42)
pls.fit(X_train, y_train)
Xf, names = pls.transform(X_test)
```

### Hybrid Composition

```python
from foodspec.features import FeatureComposer

# Combine multiple extractors
composer = FeatureComposer([
    ("pca", PCAFeatureExtractor(n_components=3), {}),
    ("pls", PLSFeatureExtractor(n_components=2), {}),
])

composer.fit(X_train, y_train)
feature_set = composer.transform(X_test)  # Returns FeatureSet

print(feature_set.Xf.shape)              # (n_test, 5)
print(feature_set.feature_names)          # ['pca_0', 'pca_1', 'pca_2', 'pls_0', 'pls_1']
```

### Stability Selection

```python
from foodspec.features import StabilitySelector

# Select stable features across bootstrap samples
selector = StabilitySelector(
    base_extractor=PCAFeatureExtractor(n_components=10),
    n_bootstrap=100,
    threshold=0.75,
    random_state=42
)

selector.fit(X_train, y_train)
Xf, names = selector.transform(X_test)
print(selector.selected_features_)  # Indices of selected features
```

## sklearn Integration

All extractors support sklearn's parameter API:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('features', PCAFeatureExtractor()),
    ('classifier', RandomForestClassifier())
])

# Grid search over feature extractor parameters
param_grid = {
    'features__n_components': [2, 3, 5],
    'classifier__n_estimators': [50, 100]
}

search = GridSearchCV(pipeline, param_grid, cv=5)
search.fit(X_train, y_train)
```

## Leakage Prevention

All extractors strictly separate fit and transform:

```python
# CORRECT: Fit only on training data
pca = PCAFeatureExtractor(n_components=3)
pca.fit(X_train)  # Learn parameters from training only
Xf_train, _ = pca.transform(X_train)
Xf_test, _ = pca.transform(X_test)  # Apply learned parameters

# WRONG: Don't fit on test data
pca.fit(X_test)  # ❌ Leakage!
```

## Determinism

For reproducible results, always set `random_state`:

```python
# Deterministic results
pca1 = PCAFeatureExtractor(n_components=3, random_state=42)
pca2 = PCAFeatureExtractor(n_components=3, random_state=42)

pca1.fit(X)
pca2.fit(X)

Xf1, _ = pca1.transform(X)
Xf2, _ = pca2.transform(X)
assert np.allclose(Xf1, Xf2)  # ✓ Identical results
```

## Migration from DataFrame API

If you have code using the old DataFrame-based API:

```python
# OLD (deprecated)
pca = PCAFeatureExtractor(n_components=3)
pca.fit(X_train)
df = pca.transform(X_test)  # Returns DataFrame
feature_values = df.values
feature_names = df.columns.tolist()

# NEW (current)
pca = PCAFeatureExtractor(n_components=3)
pca.fit(X_train)
Xf, feature_names = pca.transform(X_test)  # Returns tuple
feature_values = Xf  # Already numpy array
```

The `FeatureComposer` automatically handles both APIs for backward compatibility.

## Testing

Run feature engineering tests:

```bash
pytest tests/test_features_base.py -v      # 21 base interface tests
pytest tests/test_feature_engineering.py -v # 25 feature engineering tests
```

## Summary

- **Base class**: `FeatureExtractor` (ABC with sklearn compatibility)
- **Return format**: `(Xf, feature_names)` tuple from `transform()`
- **Container**: `FeatureSet` dataclass with numpy array and metadata
- **Validation**: Automatic shape/length checking
- **Integration**: Full sklearn GridSearchCV support via `get_params/set_params`
- **Safety**: Strict fit/transform separation prevents leakage
- **Reproducibility**: `random_state` parameter for deterministic results
