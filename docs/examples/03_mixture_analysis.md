# Mixture Analysis: Quantification via NNLS

**Level**: Intermediate → Advanced  
**Runtime**: <1 second  
**Key Concepts**: Linear spectral unmixing, NNLS, quantification, mixture models

---

## What You Will Learn

In this example, you'll learn how to:
- Create pure component spectra and synthetic mixtures
- Use Non-Negative Least Squares (NNLS) to estimate component fractions
- Assess unmixing accuracy by comparing estimated vs. actual fractions
- Understand limitations of linear mixture models
- Apply this to real-world adulterant detection

After completing this example, you'll understand how to quantify ingredient blends, detect mineral oil adulterations, and validate mixture assumptions.

---

## Prerequisites

- Understanding of linear algebra (matrix equations, least squares)
- Familiarity with optimization problems
- `numpy`, `scipy` installed
- Basic knowledge of mixture models (optional, we explain)

**Optional background**: Read [Mixture Models](../methods/chemometrics/mixture_models.md)

---

## The Problem

**Real-world scenario**: You've measured spectra for two pure oils (olive and sunflower). A customer supplies an unknown "olive oil" that might be adulterated with sunflower oil. Can you:
1. Build a mixture model from pure components?
2. Unmix the unknown to estimate its composition?
3. Assess whether the claimed olive oil is authentic?

**Assumption**: Spectra combine linearly (i.e., mixture spectrum = fraction₁ × spectrum₁ + fraction₂ × spectrum₂)

**Goal**: Recover unknown fractions by unmixing.

---

## Step 1: Create Pure Components & Mixtures

```python
import numpy as np
from scipy.optimize import nnls

# Create synthetic pure component spectra
np.random.seed(42)
n_wavelengths = 1500

# Pure component 1 (Olive oil): single broad peak
wavelengths = np.linspace(800, 3000, n_wavelengths)
pure_1 = np.exp(-((wavelengths - 1600) / 200) ** 2) + 0.05 * np.random.randn(n_wavelengths)

# Pure component 2 (Sunflower oil): different peak
pure_2 = np.exp(-((wavelengths - 1500) / 150) ** 2) + 0.05 * np.random.randn(n_wavelengths)

# Mixture: 70% olive + 30% sunflower
true_fractions = np.array([0.70, 0.30])
mixture = true_fractions[0] * pure_1 + true_fractions[1] * pure_2

print(f"Pure 1 intensity range: {pure_1.min():.3f} to {pure_1.max():.3f}")
print(f"Pure 2 intensity range: {pure_2.min():.3f} to {pure_2.max():.3f}")
print(f"Mixture composition: {true_fractions[0]*100:.0f}% Pure1 + {true_fractions[1]*100:.0f}% Pure2")
```

**What's happening**:
- `pure_1` and `pure_2`: Two reference spectra with different spectral features
- `mixture`: Linear combination of pure components with known fractions
- Noise is added to make it realistic

---

## Step 2: Unmix with NNLS

```python
# Set up linear system: A @ fractions = mixture
# where A contains pure components as columns
A = np.column_stack([pure_1, pure_2])

# Solve with Non-Negative Least Squares
estimated_fractions, residual = nnls(A, mixture)

# Normalize (fractions should sum to ~1.0)
estimated_fractions = estimated_fractions / estimated_fractions.sum()

print(f"True fractions:      {true_fractions}")
print(f"Estimated fractions: {estimated_fractions}")
print(f"Residual (RMSE):     {np.sqrt(residual):.6f}")
```

**Interpretation**:
- **NNLS output**: Estimated fractions ≥ 0 (enforces physical constraint: can't have negative amounts)
- **Residual**: How well the linear model fits (lower = better)
- **Normalized**: Fractions sum to 1.0 for direct interpretation

---

## Step 3: Assess Accuracy

```python
# Compare true vs. estimated
error = np.abs(estimated_fractions - true_fractions)
mae = error.mean()
rmse = np.sqrt((error ** 2).mean())

print(f"\nAccuracy Assessment:")
print(f"  Component 1 - True: {true_fractions[0]:.3f}, Estimated: {estimated_fractions[0]:.3f}, Error: {error[0]:.3f}")
print(f"  Component 2 - True: {true_fractions[1]:.3f}, Estimated: {estimated_fractions[1]:.3f}, Error: {error[1]:.3f}")
print(f"  Mean Absolute Error: {mae:.4f}")
print(f"  Root Mean Sq Error:  {rmse:.4f}")

# Practical decision: Is adulterant level acceptable?
adulterant_level = estimated_fractions[1] * 100
threshold = 5.0  # tolerance: up to 5% sunflower oil is acceptable
if adulterant_level > threshold:
    print(f"\n⚠️  WARNING: Adulterant detected! ({adulterant_level:.1f}% > {threshold}% threshold)")
else:
    print(f"\n✓ PASS: Adulterant level acceptable ({adulterant_level:.1f}%)")
```

**What's happening**:
- We compare estimated vs. true fractions (possible because we created synthetic data)
- Mean Absolute Error (MAE) summarizes overall accuracy
- **Real scenario**: Compare against known adulterant thresholds (regulatory limits)

---

## Step 4: Visualize Pure Components & Unmixing

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Pure components
ax = axes[0, 0]
ax.plot(wavelengths, pure_1, label="Pure Component 1 (Olive)", linewidth=2)
ax.plot(wavelengths, pure_2, label="Pure Component 2 (Sunflower)", linewidth=2)
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("Intensity")
ax.set_title("Pure Component Spectra")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Mixture spectrum
ax = axes[0, 1]
ax.plot(wavelengths, mixture, label="Unknown Mixture", linewidth=2, color="red")
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("Intensity")
ax.set_title(f"Mixture Spectrum (True: 70% Comp1, 30% Comp2)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Composition comparison
ax = axes[1, 0]
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, true_fractions, width, label="True", alpha=0.8)
ax.bar(x + width/2, estimated_fractions, width, label="Estimated", alpha=0.8)
ax.set_ylabel("Fraction")
ax.set_title("Composition: True vs. Estimated")
ax.set_xticks(x)
ax.set_xticklabels(["Component 1", "Component 2"])
ax.legend()
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis="y")

# Plot 4: Residual (fit quality)
reconstructed = A @ estimated_fractions
residuals = mixture - reconstructed
ax = axes[1, 1]
ax.plot(wavelengths, residuals, label="Unmixing Residual", linewidth=1, color="gray")
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
ax.fill_between(wavelengths, residuals, 0, alpha=0.3)
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("Residual Intensity")
ax.set_title(f"Unmixing Error (RMSE={np.sqrt(residual):.4f})")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mixture_analysis_unmixing.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Figure interpretation**:
- **Top-left**: Pure components have distinct spectral features
- **Top-right**: Mixture shows combination of both features
- **Bottom-left**: Bar chart compares true vs. estimated fractions (should match closely)
- **Bottom-right**: Residual shows unmixing error (should be small noise)

---

## Full Working Script

See the production script with multiple synthetic mixtures and detailed accuracy assessment:

📄 **[`examples/mixture_analysis_quickstart.py`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/mixture_analysis_quickstart.py)** – Full working code (55 lines)

---

## Key Takeaways

✅ **Linear mixture model**: Assumes spectra combine additively  
✅ **NNLS solver**: Enforces non-negativity (physically realistic)  
✅ **Quantification workflow**: Measure pure components → Create standard → Unmix unknown  
✅ **Validation**: Compare estimated vs. known fractions to assess accuracy  

---

## Assumptions & Limitations

⚠️ **Linear mixing**: Works for Raman/IR, not always for fluorescence or imaging
⚠️ **Known pure components**: Must measure or obtain reference spectra
⚠️ **Spectral stability**: Component spectra shouldn't change with concentration
⚠️ **No chemical reactions**: Binary mixtures are simpler than complex food matrices

---

## Real-World Applications

- 🫒 **Oil adulterant detection**: Quantify mineral oil, seed oil additions
- 🍯 **Honey verification**: Estimate corn syrup, high fructose content
- 🧈 **Butter authenticity**: Detect margarine or vegetable oil blends
- 🧂 **Salt purity**: Measure mineral contaminants
- 🍶 **Alcohol purity**: Quantify water and flavor additives

---

## Advanced Topics

**Want to go deeper?**
- **Multiple components**: Unmix > 2 components simultaneously
- **Constraints**: Add inequality bounds on fractions
- **Regularization**: Use Ridge or Lasso regression for ill-posed problems
- **Non-linear unmixing**: Handle concentration-dependent spectral shifts

See [Mixture Models](../methods/chemometrics/mixture_models.md) for complete details.

---

## Next Steps

1. **Try it**: Use actual oils or other food samples as pure components
2. **Explore**: Add measurement noise and assess robustness
3. **Learn more**: Read [Mixture Models](../methods/chemometrics/mixture_models.md)
4. **Advance**: Combine with [Oil Authentication](01_oil_authentication.md) for classification + quantification

---

## Interactive Notebook

For step-by-step exploration with multiple components:

📓 **[`examples/tutorials/03_mixture_analysis_teaching.ipynb`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/03_mixture_analysis_teaching.ipynb)**

---

## Figure provenance
- Generated by [scripts/generate_docs_figures.py](https://github.com/chandrasekarnarayana/foodspec/blob/main/scripts/generate_docs_figures.py)
- Output: [../assets/figures/cv_boxplot.png](../assets/figures/cv_boxplot.png)

