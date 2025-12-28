# Datasets API Reference

!!! info "Module Purpose"
    Bundled example datasets for testing, tutorials, and benchmarking.

---

## Quick Navigation

| Dataset | Description | Use Case |
|---------|-------------|----------|
| `load_olive_oils()` | Olive oil FTIR spectra | Classification, adulteration |
| `load_heating_study()` | Thermal degradation time series | Regression, monitoring |
| `load_mixture_data()` | Binary/ternary mixtures | MCR-ALS, mixture analysis |

---

## Common Patterns

### Pattern 1: Load Example Dataset

```python
from foodspec.data import load_olive_oils

# Load bundled dataset
fs = load_olive_oils()

print(f"Samples: {len(fs)}")
print(f"Wavenumber range: {fs.wavenumbers[0]:.0f} - {fs.wavenumbers[-1]:.0f} cm⁻¹")
print(f"Classes: {fs.metadata['variety'].unique()}")
```

### Pattern 2: Use for Quick Testing

```python
from foodspec.data import load_olive_oils
from foodspec.chemometrics import make_pls_da
from sklearn.model_selection import cross_val_score

# Quick test without loading custom data
fs = load_olive_oils()
clf = make_pls_da(n_components=5)

scores = cross_val_score(clf, fs.x, fs.metadata['variety'], cv=5)
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Available Datasets

### load_olive_oils

Olive oil FTIR spectra with variety labels.

**Description:** 120 olive oil samples across 3 varieties (extra virgin, virgin, lampante).

**Features:**
- Modality: FTIR-ATR
- Wavenumber range: 4000-650 cm⁻¹
- Classes: 3 varieties
- Samples: ~40 per class

**Example:**

```python
from foodspec.data import load_olive_oils

fs = load_olive_oils()
print(fs.metadata['variety'].value_counts())
```

---

### load_heating_study

Thermal degradation time-series data.

**Description:** Oil heating at different temperatures over time.

**Features:**
- Modality: FTIR
- Temperature: 160°C, 180°C, 200°C
- Time points: 0-8 hours
- Quality measurements included

**Example:**

```python
from foodspec.data import load_heating_study

fs = load_heating_study()
print(fs.metadata[['temperature', 'time', 'quality_score']].describe())
```

---

### load_mixture_data

Binary and ternary mixture spectra.

**Description:** Synthetic mixtures with known composition.

**Features:**
- Pure components: 3 oils
- Mixture ratios: 0-100% in 10% steps
- Ground truth concentrations

**Example:**

```python
from foodspec.data import load_mixture_data

fs = load_mixture_data()
print(f"Mixtures: {len(fs)}")
print(f"Components: {fs.metadata.columns}")
```

---

## Dataset Information

### Dataset Structure

All bundled datasets return `FoodSpectrumSet` objects with:
- `x`: Spectral data (n_samples, n_features)
- `wavenumbers`: Wavenumber array
- `metadata`: pandas DataFrame with sample information
- `modality`: Spectroscopy type (FTIR, Raman, etc.)

### Citation

If using bundled datasets in publications, cite:

```plaintext
FoodSpec Development Team. (2024). FoodSpec: Food Spectroscopy Analysis Toolkit.
https://github.com/yourusername/foodspec
```

---

## Cross-References

**Related Modules:**
- [IO](io.md) - Load custom datasets
- [Core](core.md) - Dataset structure

**Related Tutorials:**
- [Getting Started Tutorial](../tutorials/beginner/01-load-and-plot.md) - Uses olive oil dataset
- [Mixture Models (ML)](../methods/chemometrics/mixture_models.md) - Uses mixture dataset
