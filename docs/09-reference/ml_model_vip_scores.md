# Variable Importance in Projection (VIP) for PLS Models

## Overview
VIP (Variable Importance in Projection) scores quantify which spectral wavenumbers are most important for a PLS or PLS-DA model's predictions. Higher VIP scores indicate more discriminative wavelengths.

## What is VIP?

VIP is a post-hoc interpretation metric derived from the PLS weight matrix and explained variance. Mathematically:

$$\text{VIP}_j = \sqrt{\frac{p \sum_a \text{SS}_a \cdot w_{j,a}^2}{\sum_a \text{SS}_a}}$$

where:
- $p$ = number of wavenumber variables
- $a$ = PLS latent component index
- $\text{SS}_a$ = sum of squares explained by component $a$
- $w_{j,a}$ = PLS weight for wavenumber $j$ in component $a$

**Interpretation:** VIP > 1.0 suggests a wavenumber is above-average importance; VIP < 0.8 suggests below-average.

## Using VIP in FoodSpec

### Extract VIP from PLS-DA Pipeline

```python
import numpy as np
from foodspec.chemometrics.models import make_pls_da

# Train PLS-DA model
X = np.random.randn(100, 500)  # 100 samples, 500 wavenumbers
y = np.random.randint(0, 3, 100)  # 3 classes

pipeline = make_pls_da(n_components=5)
pipeline.fit(X, y)

# Extract VIP scores
pls_projector = pipeline.named_steps["pls_proj"]
vip_scores = pls_projector.get_vip_scores()

print(f"VIP shape: {vip_scores.shape}")  # (500,)
print(f"Top 10 VIP: {np.argsort(vip_scores)[::-1][:10]}")
```

### Interpretation & Visualization

```python
import matplotlib.pyplot as plt

wavenumbers = np.linspace(400, 4000, 500)  # example grid

# Plot VIP across spectrum
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(wavenumbers, vip_scores, width=2.0, alpha=0.7)
ax.axhline(y=1.0, color="red", linestyle="--", label="VIP = 1.0 threshold")
ax.axhline(y=0.8, color="orange", linestyle="--", label="VIP = 0.8 threshold")
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("VIP Score")
ax.legend()
plt.tight_layout()
plt.savefig("pls_da_vip_scores.png")
```

### Chemical Interpretation

Combine VIP with spectroscopic knowledge:

```python
from foodspec.chemometrics.validation import vip_table_with_meanings

# Get top VIP wavenumbers with chemical meanings
vip_table = vip_table_with_meanings(
    vip_scores,
    wavenumbers,
    top_n=15,
    modality="raman",
    tolerance=20.0  # ±20 cm^-1 tolerance for band matching
)

print(vip_table)
# Output:
#    wavenumber      vip       meaning
# 0        1602.3   2.145   C=C aromatic stretch
# 1        1450.1   1.892   CH2 bending
# ...
```

## Limitations & Assumptions

1. **Scale Sensitivity:** VIP depends on feature scaling. Always standardize X before PLS fitting.
   ```python
   from sklearn.preprocessing import StandardScaler
   pipeline = Pipeline([
       ("scaler", StandardScaler()),
       ("pls_da", make_pls_da(n_components=5))
   ])
   ```

2. **Multicollinearity:** Highly correlated wavenumbers may have unstable VIP scores. Use regularization.

3. **Sample Size:** Small n (< 30) → VIP unreliable. Ensure adequate replication.

4. **Number of Components:** VIP depends on `n_components`. Validate via cross-validation.
   ```python
   # Use cross-validated component count
   from sklearn.model_selection import cross_val_score
   ```

## Outputs in Protocol Results

VIP scores are automatically saved in protocol runs when using PLS-DA:

```bash
foodspec run-exp config.yml --output-dir results/
# Outputs:
# results/tables/vip_scores.csv
# results/figures/vip_barplot.png
# results/interpretation/top_vip_bands.md
```

## See Also
- [PLS Regression & PLS-DA](../03-cookbook/chemometrics_guide.md#pls-and-pls-da)
- [Model Interpretability](../ml/model_interpretability.md)
- Feature Interpretation (under review)
- [Spectroscopic Databases](../04-user-guide/vendor_io.md)
