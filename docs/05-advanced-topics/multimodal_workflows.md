# Multi-Modal & Cross-Technique Analysis

FoodSpec supports **multi-modal analysis** to combine measurements from different spectroscopic techniques (e.g., Raman + FTIR + NIR) for enhanced authentication, classification, and quality control.

## 📚 Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [API Examples](#api-examples)
  - [Creating Multi-Modal Datasets](#creating-multi-modal-datasets)
  - [Late Fusion (Feature-Level)](#late-fusion-feature-level)
  - [Decision Fusion (Prediction-Level)](#decision-fusion-prediction-level)
  - [Agreement Metrics](#agreement-metrics)
- [Use Cases](#use-cases)
- [Best Practices](#best-practices)

---

## Overview

**Why multi-modal spectroscopy?**

- **Complementary information**: Different techniques probe different molecular vibrations
- **Robustness**: Cross-technique validation reduces false positives
- **Enhanced accuracy**: Combining modalities often outperforms single techniques
- **Novelty detection**: Disagreement between modalities signals unexpected samples

**Supported workflows:**

1. **Late fusion**: Concatenate features from multiple modalities → train joint model
2. **Decision fusion**: Train separate models → combine predictions (voting/weighted)
3. **Agreement analysis**: Quantify consistency across techniques

---

## Core Concepts

### MultiModalDataset

Container for aligned spectral data across modalities.

**Key properties:**
- All modalities share the same sample IDs (aligned metadata)
- Supports subsetting, filtering, and feature extraction
- Preserves modality-specific spectral ranges

### Fusion Strategies

#### **Late Fusion** (Feature-Level)
Concatenate feature matrices before training:
```
Raman features [n×m₁] + FTIR features [n×m₂] → Joint features [n×(m₁+m₂)]
```

**Advantages:**
- Single model learns cross-modality interactions
- Lower computational cost (one model)

**Disadvantages:**
- Requires aligned data (same samples)
- May struggle with missing modalities

#### **Decision Fusion** (Prediction-Level)
Train separate models, combine predictions:
```
Raman model → predictions₁
FTIR model  → predictions₂
Combine via voting or weighted averaging
```

**Advantages:**
- Handles missing modalities gracefully
- Modality-specific tuning
- Interpretable per-modality contributions

**Disadvantages:**
- Trains multiple models (higher cost)
- May miss cross-modality interactions

---

## API Examples

### Creating Multi-Modal Datasets

```python
from foodspec.core import FoodSpectrumSet, MultiModalDataset
import numpy as np

# Load individual modalities
raman = FoodSpectrumSet.from_hdf5("raman_data.h5")
ftir = FoodSpectrumSet.from_hdf5("ftir_data.h5")

# Create multi-modal dataset (requires aligned sample IDs)
mmd = MultiModalDataset.from_datasets({
    "raman": raman,
    "ftir": ftir
})

# Access individual modalities
print(mmd.datasets["raman"].X.shape)  # (n_samples, n_raman_features)

# Subset samples (preserves alignment)
subset = mmd.subset_samples([0, 1, 2, 10, 11])

# Filter by metadata
olive_oils = mmd.filter_by_metadata(oil_type="olive")
```

### Late Fusion (Feature-Level)

```python
from foodspec.ml.fusion import late_fusion_concat
from sklearn.ensemble import RandomForestClassifier

# Extract features from each modality
feature_dict = mmd.to_feature_dict()
# {'raman': array (n, m1), 'ftir': array (n, m2)}

# Concatenate features
result = late_fusion_concat(feature_dict)
X_fused = result.X_fused  # (n, m1+m2)
boundaries = result.boundaries  # {'raman': (0, m1), 'ftir': (m1, m1+m2)}

# Train joint model
y = mmd.datasets["raman"].sample_table["label"]
clf = RandomForestClassifier()
clf.fit(X_fused, y)

# Predict on new data
y_pred = clf.predict(X_fused)
```

### Decision Fusion (Prediction-Level)

```python
from foodspec.ml.fusion import decision_fusion_vote, decision_fusion_weighted
from sklearn.svm import SVC

# Train separate models
predictions = {}
probas = {}

for modality, ds in mmd.datasets.items():
    X = ds.X
    y = ds.sample_table["label"]
    
    clf = SVC(probability=True)
    clf.fit(X, y)
    
    predictions[modality] = clf.predict(X)
    probas[modality] = clf.predict_proba(X)

# Majority voting (requires ≥50% agreement)
result_vote = decision_fusion_vote(predictions, strategy="majority")
print(result_vote.final_predictions)

# Unanimous voting (all modalities must agree)
result_unanimous = decision_fusion_vote(predictions, strategy="unanimous")
print(f"Unanimous: {result_unanimous.unanimous_fraction:.1%}")

# Weighted averaging of probabilities
weights = {"raman": 0.6, "ftir": 0.4}  # Raman more reliable
result_weighted = decision_fusion_weighted(probas, weights=weights)
y_pred_weighted = result_weighted.final_predictions
```

### Agreement Metrics

```python
from foodspec.stats.fusion_metrics import (
    modality_agreement_kappa,
    modality_consistency_rate,
    cross_modality_correlation
)

# Cohen's kappa (inter-rater agreement)
kappa_df = modality_agreement_kappa(predictions)
print(kappa_df)
#         raman  ftir
# raman    1.0   0.85
# ftir     0.85  1.0

# Consistency rate (unanimous agreement)
consistency = modality_consistency_rate(predictions)
print(f"Unanimous: {consistency:.1%}")  # e.g., 92.3%

# Cross-modality feature correlation
corr_df = cross_modality_correlation(feature_dict, method="pearson")
print(corr_df)
#         raman  ftir
# raman    1.0   0.62
# ftir     0.62  1.0
```

---

## Use Cases

### 1. Olive Oil Authentication (Raman + FTIR)

```python
# Load multi-modal data
raman = FoodSpectrumSet.from_hdf5("olive_raman.h5")
ftir = FoodSpectrumSet.from_hdf5("olive_ftir.h5")
mmd = MultiModalDataset.from_datasets({"raman": raman, "ftir": ftir})

# Late fusion approach
features = mmd.to_feature_dict()
result = late_fusion_concat(features)
X = result.X_fused
y = mmd.datasets["raman"].sample_table["authentic"]

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print(f"Late fusion accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
```

### 2. Novelty Detection via Modality Disagreement

```python
# Train separate models
from sklearn.svm import SVC

predictions = {}
for mod, ds in mmd.datasets.items():
    clf = SVC()
    clf.fit(ds.X, ds.sample_table["label"])
    predictions[mod] = clf.predict(ds.X)

# Find samples where modalities disagree
result = decision_fusion_vote(predictions, strategy="unanimous")
disagreement_idx = result.disagreement_indices

# Inspect disagreements (potential novelties or mislabels)
flagged_samples = mmd.datasets["raman"].sample_table.iloc[disagreement_idx]
print(flagged_samples[["sample_id", "label", "batch"]])
```

### 3. Robustness Validation

```python
from foodspec.stats.fusion_metrics import modality_agreement_kappa

# Compare predictions from two labs (Lab A: Raman, Lab B: FTIR)
kappa_df = modality_agreement_kappa({
    "lab_a_raman": predictions_raman,
    "lab_b_ftir": predictions_ftir
})

# Cohen's kappa interpretation:
# κ > 0.8: Excellent agreement
# 0.6–0.8: Good agreement
# 0.4–0.6: Moderate agreement
# < 0.4: Poor agreement

if kappa_df.loc["lab_a_raman", "lab_b_ftir"] > 0.8:
    print("✅ Cross-lab/cross-technique validation: EXCELLENT")
else:
    print("⚠️ Low agreement—investigate source of discrepancy")
```

---

## Best Practices

### Data Alignment

- **Ensure same sample IDs** across all modalities
- Use `sample_table` metadata to align datasets before creating `MultiModalDataset`
- Handle missing modalities explicitly (decision fusion tolerates gaps)

### Feature Scaling

- **Normalize features** before late fusion to avoid dominance by high-variance modalities
- Example:
  ```python
  from sklearn.preprocessing import StandardScaler
  
  scaler = StandardScaler()
  X_fused_scaled = scaler.fit_transform(result.X_fused)
  ```

### Model Selection

| Scenario | Recommended Strategy |
|----------|---------------------|
| All modalities always available | Late fusion |
| Missing modalities common | Decision fusion |
| Interpretability priority | Decision fusion |
| Small sample size | Decision fusion (regularization per modality) |

### Hyperparameter Tuning

- **Late fusion**: Tune single model on fused features
- **Decision fusion**: Tune each modality separately + fusion weights
- Use `boundaries` from `late_fusion_concat` for modality-specific regularization

### Validation

- **Always cross-validate** fusion strategies
- Report **per-modality performance** for transparency
- Use **agreement metrics** to diagnose fusion quality

---

## Summary

Multi-modal spectroscopy in FoodSpec enables:

✅ **Complementary information**: Raman + FTIR capture different chemical signatures  
✅ **Cross-validation**: Modality agreement confirms predictions  
✅ **Robustness**: Fusion reduces overfitting to single-technique artifacts  
✅ **Novelty detection**: Disagreement flags unexpected samples  

**Next steps:**
- See [Quickstart: Python API](../01-getting-started/quickstart_python.md) for dataset basics
- See [Chemometrics Guide](../03-cookbook/chemometrics_guide.md) for model selection
- See [Validation Protocols](../03-cookbook/validation_chemometrics_oils.md) for cross-validation best practices
