# Feature Engineering for FoodSpec v2

Comprehensive feature extraction system for spectral data analysis with leakage-safe design, deterministic outputs, and hybrid composition support.

## Overview

The FoodSpec feature engineering system provides:

- **Peak-based features**: Heights, areas, and ratios at specified wavenumbers
- **Band integration**: Spectral region areas with optional baseline subtraction
- **Chemometric embeddings**: PCA (unsupervised) and PLS (supervised) dimensionality reduction
- **Hybrid composition**: Chain multiple extractors for multi-modal feature sets
- **Feature selection**: Stability selection for minimal marker panels
- **Leakage safety**: Strict fit/transform separation ensures training-only fitting
- **Determinism**: Reproducible results with explicit seeds

## Core Abstractions

### FeatureExtractor Protocol

All extractors implement a consistent interface:

```python
class FeatureExtractor(Protocol):
    def fit(self, X, y=None, **kwargs) -> FeatureExtractor:
        """Fit on training data only."""
        ...
    
    def transform(self, X, **kwargs) -> pd.DataFrame:
        """Transform fitted data to features."""
        ...
    
    def fit_transform(self, X, y=None, **kwargs) -> pd.DataFrame:
        """Convenience: fit and transform in one call."""
        ...
```

**Leakage Safety**: `fit()` MUST only touch training data; `transform()` applies to test/validation.

### FeatureSet Container

Structured output with metadata:

```python
@dataclass
class FeatureSet:
    features: pd.DataFrame          # Extracted features
    feature_names: List[str]        # Column names
    extractor_names: List[str]      # Source extractors
    metadata: Dict[str, Any]        # Explained variance, etc.
```

## Feature Extractors

### 1. Peak-Based Features

#### PeakHeights

Extract maximum intensity values at specified wavenumbers:

```python
from foodspec.features import PeakHeights

extractor = PeakHeights(
    peak_locations=[1030, 1050, 1200],  # Target wavenumbers
    window=15                             # Search window (±15 indices)
)

features = extractor.compute(X, x)  # Returns DataFrame with peak_height_* columns
```

**Use cases**: Quantify specific functional group intensities (e.g., carbonyl C=O at 1700 cm⁻¹)

#### PeakAreas

Integrate intensity around peaks with optional baseline subtraction:

```python
from foodspec.features import PeakAreas

extractor = PeakAreas(
    peak_locations=[1500, 1700],
    window=20,
    baseline_subtract=True  # Linear baseline from endpoints
)

features = extractor.compute(X, x)  # Returns peak_area_* columns
```

**Use cases**: Total peak areas for quantitative analysis, remove baseline drift

#### PeakRatios

Compute ratios between peak pairs:

```python
from foodspec.features import PeakRatios

extractor = PeakRatios(
    peak_pairs=[(1030, 1050), (1200, 1600)],  # (numerator, denominator) pairs
    window=15,
    eps=1e-12  # Avoid division by zero
)

features = extractor.compute(X, x)  # Returns ratio_1030_1050, ratio_1200_1600
```

**Use cases**: Normalize for concentration-independent features, ratiometric biomarkers

### 2. Band Integration

Compute integrated areas over spectral regions:

```python
from foodspec.features import BandIntegration

extractor = BandIntegration(
    bands=[
        (1400, 1600),  # Protein amide I region
        (2800, 3000),  # C-H stretching region
    ],
    baseline_subtract=False  # Optional linear baseline removal
)

features = extractor.compute(X, x)  # Returns band_1400_1600, band_2800_3000
```

**Use cases**: Quantify broad functional group contributions, compositional analysis

### 3. Chemometric Embeddings

#### PCA (Unsupervised)

Principal components for dimensionality reduction:

```python
from foodspec.features import PCAFeatureExtractor

pca = PCAFeatureExtractor(
    n_components=5,
    whiten=False,
    random_state=42  # Deterministic
)

# Fit on training data ONLY
pca.fit(X_train)

# Transform train and test
features_train = pca.transform(X_train)  # (n_train, 5)
features_test = pca.transform(X_test)    # (n_test, 5)

# Check explained variance
print(pca.explained_variance_ratio_)  # [0.45, 0.23, 0.12, ...]
```

**Use cases**: Data compression, visualization, remove collinearity

#### PLS (Supervised)

Partial least squares with label information:

```python
from foodspec.features import PLSFeatureExtractor

pls = PLSFeatureExtractor(
    n_components=3,
    scale=True,  # Scale to unit variance
    random_state=42
)

# Fit requires labels
pls.fit(X_train, y_train)

# Transform test data
features_test = pls.transform(X_test)  # (n_test, 3)
```

**Use cases**: Supervised dimensionality reduction, maximize class separation

### 4. Hybrid Composition

Chain multiple extractors for multi-modal features:

```python
from foodspec.features import FeatureComposer, PCAFeatureExtractor, PeakHeights, BandIntegration

composer = FeatureComposer([
    ("pca", PCAFeatureExtractor(n_components=5), {}),
    ("peaks", PeakHeights([1200, 1500, 1700]), {"x": x_grid}),
    ("bands", BandIntegration([(1400, 1600)]), {"x": x_grid}),
])

# Fit all extractors on training data
composer.fit(X_train, y_train, x=x_grid)

# Transform returns FeatureSet
feature_set = composer.transform(X_test, x=x_grid)

print(feature_set.features.shape)        # (n_test, 5+3+1)
print(feature_set.feature_names)         # ['pc1', 'pc2', ..., 'peak_height_1200', ..., 'band_1400_1600']
print(feature_set.extractor_names)       # ['pca', 'pca', ..., 'peaks', ..., 'bands']
```

**Use cases**: Combine complementary feature types, improved classification performance

### 5. Stability Selection for Marker Panels

Select minimal feature sets via repeated subsampling:

```python
from foodspec.features import StabilitySelector
from foodspec.models import LogisticRegressionClassifier

selector = StabilitySelector(
    estimator_factory=lambda: LogisticRegressionClassifier(penalty="l1", C=1.0, solver="saga"),
    n_resamples=50,       # Bootstrap iterations
    subsample_fraction=0.7,
    selection_threshold=0.5,  # Min frequency to keep feature
    random_state=42
)

# Fit on training data ONLY
selector.fit(X_train, y_train)

# Transform to selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Get marker panel metadata
marker_panel = selector.get_marker_panel(x_wavenumbers=x_grid)
print(marker_panel["selected_indices"])       # [5, 12, 45, 67]
print(marker_panel["selected_wavenumbers"])   # [1025.3, 1456.7, ...]
print(marker_panel["selection_frequencies"])  # [0.92, 0.78, 0.65, ...]
```

**Use cases**: Minimal biomarker panels, interpretability, reduce overfitting

## Integration with Cross-Validation

**Critical**: Extractors MUST be fit inside CV loops to avoid leakage:

```python
from foodspec.validation import EvaluationRunner
from foodspec.models import LogisticRegressionClassifier

# Example: PCA + Classification in CV
for train_idx, test_idx in cv_splitter.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Fit PCA on training fold ONLY
    pca = PCAFeatureExtractor(n_components=5)
    pca.fit(X_train)
    
    # Transform both splits
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Train and evaluate model
    clf = LogisticRegressionClassifier()
    clf.fit(X_train_pca.values, y_train)
    predictions = clf.predict_proba(X_test_pca.values)
```

**Leakage prevention verified by tests**: See `test_feature_engineering.py::TestLeakageSafety`

## Determinism

All extractors with randomization accept `random_state` for reproducibility:

```python
# Same seed = same results
pca1 = PCAFeatureExtractor(n_components=3, random_state=42)
pca1.fit(X)
features1 = pca1.transform(X_test)

pca2 = PCAFeatureExtractor(n_components=3, random_state=42)
pca2.fit(X)
features2 = pca2.transform(X_test)

np.testing.assert_array_almost_equal(features1.values, features2.values)  # Pass
```

## Input/Output Conventions

### Standard Signature

All extractors accept:

- **Input**: `X` (2D numpy array: samples × features), optional `x` (1D wavenumber grid), optional `y` (labels for supervised methods)
- **Output**: `pd.DataFrame` with named columns

### Feature Naming

- Peak heights: `peak_height_<wavenumber>`
- Peak areas: `peak_area_<wavenumber>`
- Peak ratios: `ratio_<w1>_<w2>`
- Band integration: `band_<start>_<end>`
- PCA: `pc1`, `pc2`, ...
- PLS: `pls1`, `pls2`, ...

## Testing

Comprehensive test suite (`tests/test_feature_engineering.py`):

- ✅ All extractors produce expected output shapes
- ✅ Leakage safety: fit only touches training data
- ✅ Determinism: same seed → same features
- ✅ Protocol conformance: all implement FeatureExtractor
- ✅ Input validation: actionable errors for invalid inputs
- ✅ Hybrid composition: correct feature concatenation

Run tests:

```bash
pytest tests/test_feature_engineering.py -v
```

## Best Practices

1. **Always fit on training data only**: Never fit extractors on full dataset before CV
2. **Use consistent seeds**: Enable reproducibility in publications
3. **Validate inputs**: Check X and x shapes match before extraction
4. **Document feature choices**: Record wavenumber selections and rationale
5. **Start simple**: Begin with peak-based features, add chemometric embeddings if needed
6. **Combine complementary features**: Use FeatureComposer for multi-modal analysis
7. **Apply selection cautiously**: Stability selection reduces features but requires many resamples

## Performance Tips

- **Peak extraction scales linearly** with number of peaks
- **PCA/PLS**: Use `n_components` < 20 for typical spectral data
- **Band integration**: Fewer, wider bands are faster than many narrow bands
- **Stability selection**: Computationally expensive (n_resamples × CV folds); start with n_resamples=20-50

## Module Structure

```
foodspec/features/
├── base.py           # FeatureExtractor protocol, FeatureSet container
├── peaks.py          # PeakHeights, PeakAreas, PeakRatios
├── bands.py          # BandIntegration
├── chemometrics.py   # PCAFeatureExtractor, PLSFeatureExtractor
├── composer.py       # FeatureComposer for hybrid features
└── selection.py      # StabilitySelector for marker panels
```

## Examples

See comprehensive usage examples in:

- `tests/test_feature_engineering.py` - All feature extractors with real data
- `foodspec/features/__init__.py` - Quick-start snippets
- Individual module docstrings - Detailed API documentation

## References

- **PCA**: Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
- **PLS**: Wold, S., et al. (2001). "PLS-regression: a basic tool of chemometrics". *Chemometrics and Intelligent Laboratory Systems*.
- **Stability Selection**: Meinshausen, N., & Bühlmann, P. (2010). "Stability selection". *Journal of the Royal Statistical Society: Series B*.
