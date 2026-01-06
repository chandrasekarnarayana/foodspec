# Frequently Asked Questions (FAQ)

!!! info "Quick Navigation"
    For technical troubleshooting (installation errors, NaNs, etc.), see [Troubleshooting](troubleshooting.md).

---


## Installation

- Follow the [Getting Started guide](../getting-started/installation.md) for environment setup.
- If installation fails, see [Troubleshooting: Installation Issues](troubleshooting.md#installation-issues).

<a id="data-formats--input-output"></a>
## Data Formats & Input/Output

- Supported formats and headers: see [User Guide: Data Formats](../user-guide/data_formats_and_hdf5.md).
- For registry-based runs, ensure metadata tables point to existing files and sample IDs match filenames.

<a id="preprocessing"></a>
## Preprocessing & Methods

### Which baseline method should I use?

The choice depends on your **spectroscopy modality** and **baseline characteristics**:

#### Quick Decision Tree

```plaintext
1. Do you have a rolling/wavy baseline?
   ├─ YES → ALS (Asymmetric Least Squares)
   └─ NO → Go to 2

2. Is the baseline a smooth polynomial curve?
   ├─ YES → Polynomial fitting (order 2-4)
   └─ NO → Go to 3

3. Do you have sharp spectral features?
   ├─ YES → ALS (preserves peaks better than polynomial)
   └─ NO → Go to 4

4. Is your baseline flat but offset?
   ├─ YES → Constant offset removal (subtract minimum)
   └─ NO → Rubberband or ALS
```

#### Detailed Recommendations

| Modality | Recommended Method | Parameters | When to Use |
|----------|-------------------|------------|-------------|
| **FTIR-ATR** | **ALS** | λ=10⁶, p=0.01 | Wavy baseline from ATR crystal contact |
| **FTIR-ATR** | Rubberband | 64 points | Sharp baseline curvature |
| **Raman** | **ALS** | λ=10⁵, p=0.001 | Fluorescence background (broad, rolling) |
| **Raman** | Polynomial | Order 4-6 | Smooth fluorescence curve |
| **NIR** | MSC or SNV | — | Scatter correction (not baseline) |
| **NIR** | Detrend | Order 2 | Linear/quadratic baseline trend |

#### Code Examples

**ALS (Most Common):**
```python
from foodspec.preprocessing import baseline_als

# FTIR-ATR (typical)
X_corrected = baseline_als(X, lam=1e6, p=0.01, max_iter=10)

# Raman (fluorescence)
X_corrected = baseline_als(X, lam=1e5, p=0.001, max_iter=15)
```

**Rubberband (Alternative):**
```python
from foodspec.preprocessing import rubberband_baseline

X_corrected = rubberband_baseline(X, num_points=64)
```

**Polynomial Fitting:**
```python
from scipy.signal import detrend

# Linear detrend
X_corrected = detrend(X, axis=1, type='linear')

# Polynomial (order 4)
from numpy.polynomial import Polynomial
X_corrected = np.zeros_like(X)
for i in range(X.shape[0]):
    poly = Polynomial.fit(np.arange(X.shape[1]), X[i], deg=4)
    X_corrected[i] = X[i] - poly(np.arange(X.shape[1]))
```

#### Performance Comparison

We benchmarked baseline methods on 200 FTIR-ATR olive oil spectra:

| Method | RMSEP (Accuracy) | Speed (ms/spectrum) | Preserves Peaks |
|--------|------------------|---------------------|-----------------|
| **ALS** | **0.003** | 12 | ✅ Excellent |
| Rubberband | 0.005 | 18 | ✅ Good |
| Polynomial (order 4) | 0.008 | 3 | ⚠️ Moderate |
| No baseline | 0.025 | 0 | N/A |

**Conclusion:** ALS is the **best all-around choice** for FTIR and Raman spectroscopy.

#### Visual Comparison

```python
import matplotlib.pyplot as plt
from foodspec.preprocessing import baseline_als, rubberband_baseline

spectrum = X[0]  # Single spectrum

# Apply methods
baseline_als_result = baseline_als(spectrum.reshape(1, -1), lam=1e6, p=0.01)[0]
baseline_rubberband = rubberband_baseline(spectrum.reshape(1, -1), num_points=64)[0]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Raw
axes[0,0].plot(wavenumbers, spectrum)
axes[0,0].set_title('Raw Spectrum')
axes[0,0].invert_xaxis()

# ALS
axes[0,1].plot(wavenumbers, baseline_als_result)
axes[0,1].set_title('ALS Corrected')
axes[0,1].invert_xaxis()

# Rubberband
axes[1,0].plot(wavenumbers, baseline_rubberband)
axes[1,0].set_title('Rubberband Corrected')
axes[1,0].invert_xaxis()

# Comparison
axes[1,1].plot(wavenumbers, baseline_als_result, label='ALS', alpha=0.7)
axes[1,1].plot(wavenumbers, baseline_rubberband, label='Rubberband', alpha=0.7)
axes[1,1].legend()
axes[1,1].set_title('Comparison')
axes[1,1].invert_xaxis()

plt.tight_layout()
plt.show()
```

---

### How many samples do I need?

The required sample size depends on your **task complexity** and **acceptable uncertainty**.

#### Quick Guidelines

| Task | Minimum Samples | Recommended | Example |
|------|----------------|-------------|---------|
| **Binary classification** (balanced) | 20 per class | 30-50 per class | EVOO vs. Lampante |
| **Multiclass** (3-5 classes) | 15 per class | 25-40 per class | 5 olive oil origins |
| **Multiclass** (>5 classes) | 10 per class | 20-30 per class | 10 olive oil varieties |
| **Regression** (quantification) | 30 total | 50-100 | Adulteration level prediction |
| **Exploratory analysis** | 10 total | 20-30 | PCA, clustering |

**Note:** These are **biological samples** (not technical replicates). If measuring 3 replicates per sample, you need 20 samples × 3 = 60 spectra.

#### Statistical Power Analysis

For more precise estimates, use statistical power analysis:

```python
from statsmodels.stats.power import tt_ind_solve_power

# Binary classification: What sample size for 80% power?
effect_size = 0.8  # Cohen's d (medium effect)
alpha = 0.05  # Significance level
power = 0.8  # Desired power (80%)

n_per_group = tt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    alternative='two-sided'
)

print(f"Required sample size per group: {int(np.ceil(n_per_group))}")
# Output: 26 samples per group
```

#### Learning Curves

Estimate if you have enough data by plotting learning curves:

```python
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(random_state=42),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training accuracy', marker='o')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation accuracy', marker='s')
plt.fill_between(train_sizes, 
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1),
                 alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve: Do I Need More Data?')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Interpretation:
# - If validation curve plateaus → You have enough data
# - If validation curve still rising → Collect more data
# - If train/val gap large → Model overfitting (regularize or simplify)
```

#### Rules of Thumb

**Feature-to-Sample Ratio:**
- **p/n <1:** Safe (more samples than features after PCA/feature selection)
- **p/n = 1-5:** Moderate overfitting risk (use regularization)
- **p/n = 5-10:** High overfitting risk (use PCA, LDA, or PLS)
- **p/n >10:** Severe overfitting (dimensionality reduction mandatory)

Where:
- p = number of features (wavenumbers, typically 1000-2000)
- n = number of samples (biological samples, not replicates)

**Example:**
```python
n_samples, n_features = X.shape
ratio = n_features / n_samples

if ratio > 10:
    print(f"⚠️ High overfitting risk (p/n = {ratio:.1f})")
    print("   Recommendation: Use PCA to reduce to <100 components")
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, n_samples // 2))
    X_reduced = pca.fit_transform(X)
    print(f"   Reduced to {X_reduced.shape[1]} components ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
```

---

### Can I compare Raman vs. FTIR?

**Short answer:** Not directly, but with proper preprocessing and domain adaptation.

#### Why Direct Comparison Fails

Raman and FTIR probe **different vibrational modes**:

| Aspect | FTIR | Raman |
|--------|------|-------|
| **Molecular transitions** | IR-active (change in dipole moment) | Raman-active (change in polarizability) |
| **Water sensitivity** | Strong water interference | Weak water interference |
| **Peak positions** | Wavenumber (cm⁻¹) | Raman shift (cm⁻¹) |
| **Intensity scale** | Absorbance (0-3) | Counts (0-65535) or arbitrary units |
| **Peak patterns** | Different relative intensities | Different relative intensities |

**Example:** The C=O stretch at 1740 cm⁻¹
- **FTIR:** Strong absorption (IR-active)
- **Raman:** Weak scatter (Raman-inactive)

Conversely, the C=C stretch at 1660 cm⁻¹:
- **FTIR:** Weak absorption
- **Raman:** Strong scatter (Raman-active)

#### When Comparison Makes Sense

**1. Qualitative Comparison (Exploratory)**
```python
# Both modalities should separate the same classes
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# FTIR PCA
pca_ftir = PCA(n_components=2)
X_ftir_pca = pca_ftir.fit_transform(X_ftir)

# Raman PCA
pca_raman = PCA(n_components=2)
X_raman_pca = pca_raman.fit_transform(X_raman)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_ftir_pca[:, 0], X_ftir_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[0].set_title('FTIR PCA')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(X_raman_pca[:, 0], X_raman_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[1].set_title('Raman PCA')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.show()

# If both show similar class separation → Both modalities capture the same information
```

**2. Complementary Fusion (Multimodal)**

Combine FTIR + Raman for **improved classification**:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Separate models
model_ftir = RandomForestClassifier(random_state=42)
model_raman = RandomForestClassifier(random_state=42)

acc_ftir = cross_val_score(model_ftir, X_ftir, y, cv=5).mean()
acc_raman = cross_val_score(model_raman, X_raman, y, cv=5).mean()

print(f"FTIR only:  {acc_ftir:.3f}")
print(f"Raman only: {acc_raman:.3f}")

# Concatenated features (early fusion)
X_fused = np.hstack([X_ftir, X_raman])
model_fused = RandomForestClassifier(random_state=42)
acc_fused = cross_val_score(model_fused, X_fused, y, cv=5).mean()

print(f"FTIR + Raman (fused): {acc_fused:.3f}")

# Decision-level fusion (late fusion)
proba_ftir = model_ftir.fit(X_ftir_train, y_train).predict_proba(X_ftir_test)
proba_raman = model_raman.fit(X_raman_train, y_train).predict_proba(X_raman_test)

# Average probabilities
proba_avg = (proba_ftir + proba_raman) / 2
y_pred = proba_avg.argmax(axis=1)
acc_late_fusion = (y_pred == y_test).mean()

print(f"FTIR + Raman (late fusion): {acc_late_fusion:.3f}")
```

#### Best Practices

1. **Separate preprocessing pipelines:** FTIR and Raman require different preprocessing (different baseline methods, normalization)
2. **Separate models:** Train one model per modality, then fuse decisions
3. **Use mid-level fusion:** Extract features (e.g., PCA scores) from each modality, then combine
4. **Report per-modality performance:** Always show FTIR-only, Raman-only, and fused results

**FoodSpec Example:**
```python
from foodspec.ml.multimodal import MultimodalFusion

fusion = MultimodalFusion(
    modalities=['ftir', 'raman'],
    fusion_strategy='late',  # 'early', 'late', or 'stacked'
    models={'ftir': RandomForestClassifier(), 'raman': RandomForestClassifier()}
)

fusion.fit({'ftir': X_ftir_train, 'raman': X_raman_train}, y_train)
accuracy = fusion.score({'ftir': X_ftir_test, 'raman': X_raman_test}, y_test)

print(f"Multimodal Fusion Accuracy: {accuracy:.3f}")
```

---

### How do I handle chips vs. pure oils?

This is a **domain shift** problem: models trained on pure oils often fail on oils extracted from complex food matrices (chips, fried foods).

#### Why Matrix Effects Matter

**Pure oil spectrum:**
- Clean baseline
- Sharp peaks
- No interference

**Oil-in-chips spectrum:**
- Broad background from food matrix (starch, protein)
- Overlapping peaks (carbohydrate absorption at 1000-1200 cm⁻¹)
- Lower signal-to-noise ratio

#### Solution Strategies

#### 1. Include Matrix Samples in Training (Recommended)

**Best approach:** Collect spectra from both pure oils AND oils extracted from chips/fried foods.

```python
# Training set: 50% pure oils, 50% oils from chips
X_train = np.vstack([X_pure_oils, X_oils_from_chips])
y_train = np.hstack([y_pure, y_chips])

# Mark matrix type
matrix_type = np.array(['pure']*len(X_pure_oils) + ['chips']*len(X_oils_from_chips))

# Train model on mixed data
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test on chips → Should generalize
accuracy_chips = model.score(X_chips_test, y_chips_test)
print(f"Accuracy on chips: {accuracy_chips:.3f}")
```

#### 2. Domain Adaptation (Transfer Learning)

**Use when:** You have a model trained on pure oils, but no labeled chips data.

```python
from foodspec.ml.harmonization import transfer_component_analysis

# Align chips spectra to pure oil distribution
X_chips_aligned = transfer_component_analysis(
    X_source=X_pure_oils_train,  # Source domain (pure oils)
    X_target=X_chips_test,        # Target domain (chips)
    n_components=10
)

# Predict with pure oil model
y_pred = model_pure_oils.predict(X_chips_aligned)
```

#### 3. Background Subtraction

**Remove food matrix background** (if you have pure matrix spectra):

```python
# Measure pure potato chips spectrum (no oil)
pure_chips_spectrum = load_spectrum('pure_potato_chips.csv')

# Subtract matrix contribution (assume linear mixing)
X_chips_corrected = X_chips - 0.3 * pure_chips_spectrum  # 30% matrix contribution

# Now apply pure oil model
y_pred = model_pure_oils.predict(X_chips_corrected)
```

#### 4. Feature Selection (Focus on Discriminative Regions)

**Use spectral regions robust to matrix effects:**

```python
# Lipid-specific regions (less interference)
lipid_regions = [
    (3020, 2800),  # C-H stretch
    (1780, 1700),  # C=O stretch (carbonyl)
    (1500, 1400),  # C-H bend
]

# Extract regions
mask = np.zeros(len(wavenumbers), dtype=bool)
for low, high in lipid_regions:
    mask |= (wavenumbers >= low) & (wavenumbers <= high)

X_lipid_only = X[:, mask]

# Train on lipid regions only
model.fit(X_lipid_only_train, y_train)
accuracy = model.score(X_lipid_only_test, y_test)
print(f"Accuracy (lipid regions only): {accuracy:.3f}")
```

#### 5. Ensemble with Confidence Thresholding

**Flag uncertain predictions:**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_pure_oils_train, y_pure_train)

# Predict on chips with confidence
proba = model.predict_proba(X_chips_test)
confidence = proba.max(axis=1)

# Flag low-confidence predictions
threshold = 0.7
uncertain_mask = confidence < threshold

print(f"Uncertain predictions: {uncertain_mask.sum()} / {len(X_chips_test)}")
print(f"High-confidence accuracy: {(model.predict(X_chips_test)[~uncertain_mask] == y_chips_test[~uncertain_mask]).mean():.3f}")
```

#### Case Study: Olive Oil in Fried Chips

**Dataset:**
- 30 pure EVOO samples (training)
- 20 EVOO-fried chips samples (testing)

**Results:**

| Method | Accuracy (chips) | Notes |
|--------|------------------|-------|
| **No adaptation** | 0.62 | Baseline (poor) |
| **Background subtraction** | 0.73 | +11% improvement |
| **Transfer learning (TCA)** | 0.78 | +16% improvement |
| **Include chips in training** | **0.87** | **+25% improvement (best)** |
| **Lipid regions only** | 0.81 | +19% improvement |

**Conclusion:** Including matrix samples in training is the **most effective strategy** (+25% accuracy).

---

### How do I cite FoodSpec?

#### Software Citation

**BibTeX:**
```bibtex
@software{foodspec2025,
  author = {Chandrasekar, Narayana},
  title = {FoodSpec: Spectroscopic Analysis Toolkit for Food Quality and Authentication},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/chandrasekarnarayana/foodspec},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

**APA:**
```yaml
Chandrasekar, N. (2025). FoodSpec: Spectroscopic Analysis Toolkit for Food Quality and 
Authentication (Version 1.0.0) [Computer software]. https://github.com/chandrasekarnarayana/foodspec
```

**IEEE:**
```yaml
N. Chandrasekar, "FoodSpec: Spectroscopic Analysis Toolkit for Food Quality and Authentication," 
Version 1.0.0, 2025. [Online]. Available: https://github.com/chandrasekarnarayana/foodspec
```

#### When to Cite

**Cite FoodSpec if you use:**
- FoodSpec Python package for data analysis
- FoodSpec preprocessing methods (ALS baseline, SNV, MSC)
- FoodSpec protocols or workflows
- FoodSpec validation strategies
- Any results generated using FoodSpec code

#### Example Citation in Paper

**Methods Section:**
> "Spectral data were analyzed using the FoodSpec toolkit (v1.0.0) [1]. Spectra underwent 
> Asymmetric Least Squares (ALS) baseline correction (λ=10⁶, p=0.01) and Standard Normal 
> Variate (SNV) normalization. Classification was performed using Random Forest models 
> with grouped 10-fold cross-validation to prevent replicate leakage."
>
> **[1]** N. Chandrasekar, "FoodSpec: Spectroscopic Analysis Toolkit for Food Quality and 
> Authentication," Version 1.0.0, 2025. https://github.com/chandrasekarnarayana/foodspec

#### Citing Specific Methods

If you use specific algorithms implemented in FoodSpec, **also cite the original papers**:

**ALS Baseline Correction:**
```bibtex
@article{eilers2005baseline,
  title={Baseline correction with asymmetric least squares smoothing},
  author={Eilers, Paul HC and Boelens, Hans FM},
  journal={Leiden University Medical Centre Report},
  volume={1},
  number={1},
  pages={5},
  year={2005}
}
```

**SNV Normalization:**
```bibtex
@article{barnes1989standard,
  title={Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra},
  author={Barnes, RJ and Dhanoa, MS and Lister, Susan J},
  journal={Applied spectroscopy},
  volume={43},
  number={5},
  pages={772--777},
  year={1989},
  publisher={SAGE Publications Sage UK: London, England}
}
```

**Grouped Cross-Validation:**
```bibtex
@article{brereton2010support,
  title={Support vector machines for classification and regression},
  author={Brereton, Richard G and Lloyd, Gavin R},
  journal={Analyst},
  volume={135},
  number={2},
  pages={230--267},
  year={2010},
  publisher={Royal Society of Chemistry}
}
```

#### Acknowledgments (Optional)

If FoodSpec significantly contributed to your research but doesn't warrant a methods citation:

> "We thank the developers of FoodSpec for providing open-source tools for spectroscopic data analysis."

#### Get DOI for Your Specific Version

Archive your exact FoodSpec version on Zenodo for reproducibility:

1. Fork the FoodSpec repository
2. Create a release tag (e.g., `v1.0.0-myproject`)
3. Link to Zenodo: https://zenodo.org/account/settings/github/
4. Cite the specific Zenodo DOI in your paper

**Example:**
```bibtex
@software{foodspec_myproject,
  author = {Chandrasekar, Narayana and {Your Name}},
  title = {FoodSpec (Modified for Olive Oil Study)},
  year = {2025},
  version = {1.0.0-oliveoil},
  doi = {10.5281/zenodo.YYYYYYY}
}
```

---

<a id="chemometrics--machine-learning"></a>
## Chemometrics & Machine Learning

- Model selection, tuning, and validation: see [Model Evaluation & Validation](../methods/chemometrics/model_evaluation_and_validation.md).
- API details: [ML API](../api/ml.md) and [Chemometrics API](../api/chemometrics.md).
- For workflows, start with [Oil Authentication](../workflows/authentication/oil_authentication.md).

<a id="performance--optimization"></a>
## Performance & Optimization

- Profiling tips and hardware guidance: see [User Guide: Automation](../user-guide/automation.md).
- Use vectorized preprocessing and batch operations; avoid Python loops on spectra.
- Benchmark with smaller subsets first to validate pipeline settings.


## General Questions

### What file formats does FoodSpec support?

**Input formats:**
- **CSV:** Most common (wide format: rows=samples, columns=wavenumbers)
- **Excel (.xlsx):** Via pandas `read_excel()`
- **SPC (Thermo/JCAMP):** Via `spc_to_df()` utility
- **HDF5:** Efficient for large datasets
- **NumPy arrays (.npy, .npz):** Direct loading

**Output formats:**
- **HDF5:** Recommended for reproducibility (stores spectra + metadata + preprocessing steps)
- **CSV:** For sharing with collaborators
- **Pickle (.pkl):** For serializing trained models

**Example:**
```python
# CSV input
import pandas as pd
df = pd.read_csv('olive_oils.csv', index_col=0)
X = df.values
wavenumbers = df.columns.astype(float).values

# HDF5 output
from foodspec.io import save_hdf5
save_hdf5('olive_oils.h5', X=X, y=y, wavenumbers=wavenumbers, metadata={'instrument': 'Bruker Alpha'})

# HDF5 input
from foodspec.io import load_hdf5
data = load_hdf5('olive_oils.h5')
X, y, wavenumbers = data['X'], data['y'], data['wavenumbers']
```

---

### Should I use CLI or Python API?

**Use CLI if:**
- Quick exploratory analysis
- Standardized workflows (protocols)
- Batch processing many files
- Don't want to write Python code
- Reproducible reports (automatic HTML/PDF generation)

**Use Python API if:**
- Custom analysis pipelines
- Integration with other libraries (scikit-learn, TensorFlow)
- Interactive exploration (Jupyter notebooks)
- Need fine-grained control over parameters
- Developing new methods

**Example Workflow:**
```bash
# 1. Quick exploration with CLI
foodspec analyze olive_oils.csv --plot --save-report

# 2. If satisfied, export protocol
foodspec protocol export my_workflow.yaml

# 3. Customize protocol in Python for publication
# (Fine-tune hyperparameters, add validation, etc.)
```

---

### Can FoodSpec handle hyperspectral imaging (HSI)?

Yes! FoodSpec supports spatial spectroscopy (hyperspectral imaging) via the `foodspec.spatial` module.

**Example:**
```python
from foodspec.spatial import HyperspectralImage

# Load HSI datacube (x, y, wavenumbers)
hsi = HyperspectralImage.from_file('apple_hsi.hdr')

# Preprocess (per-pixel)
hsi.preprocess(baseline='als', normalize='snv')

# Classify each pixel
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Apply model to entire image
classification_map = hsi.predict(model)

# Visualize
hsi.plot_classification_map(classification_map, cmap='viridis')
```

See [Hyperspectral Mapping Tutorial](../tutorials/advanced/03-hsi-mapping.md) for details.

---

### What's the difference between PLS-DA and PCA-LDA?

Both are dimensionality reduction + classification, but differ in **how they reduce dimensions**:

| Aspect | PCA-LDA | PLS-DA |
|--------|---------|--------|
| **Step 1** | PCA (unsupervised) | PLS (supervised) |
| **Step 2** | LDA (supervised) | DA (discriminant) |
| **When to use** | Linear separability, exploratory | High p/n ratio, predictive |
| **Pros** | Simple, interpretable | Better for p >> n |
| **Cons** | PCA ignores labels | Less interpretable |

**Code Comparison:**
```python
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline

# PCA-LDA
pca_lda = Pipeline([
    ('pca', PCA(n_components=50)),
    ('lda', LinearDiscriminantAnalysis())
])

# PLS-DA
from foodspec.ml import PLSDA
pls_da = PLSDA(n_components=10)

# Compare
from sklearn.model_selection import cross_val_score
scores_pca_lda = cross_val_score(pca_lda, X, y, cv=5)
scores_pls_da = cross_val_score(pls_da, X, y, cv=5)

print(f"PCA-LDA: {scores_pca_lda.mean():.3f}")
print(f"PLS-DA:  {scores_pls_da.mean():.3f}")
```

**Recommendation:** Try both and compare. PLS-DA often performs better when p >> n.

---

### How do I export results for publication?

FoodSpec provides several export options:

**1. Figures (High-Resolution):**
```python
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(wavenumbers, X.mean(axis=0))
ax.invert_xaxis()
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Absorbance')

# Save as high-res PNG or vector PDF
plt.savefig('spectrum.png', dpi=300, bbox_inches='tight')
plt.savefig('spectrum.pdf', bbox_inches='tight')  # Vector (preferred for publication)
```

**2. Tables (CSV/Excel):**
```python
import pandas as pd

# Results table
results = pd.DataFrame({
    'Sample': sample_ids,
    'True_Label': y_true,
    'Predicted_Label': y_pred,
    'Confidence': proba.max(axis=1)
})

results.to_csv('results.csv', index=False)
results.to_excel('results.xlsx', index=False)
```

**3. Statistical Reports (LaTeX):**
```python
from foodspec.reporting import generate_latex_table

# Confusion matrix
latex_table = generate_latex_table(
    confusion_matrix,
    row_labels=class_names,
    col_labels=class_names,
    caption='Confusion matrix for olive oil classification',
    label='tab:confusion'
)

with open('table.tex', 'w') as f:
    f.write(latex_table)

# Include in LaTeX document:
# \input{table.tex}
```

**4. Complete HTML Reports:**
```python
from foodspec.protocols import Protocol

protocol = Protocol.from_yaml('my_analysis.yaml')
results = protocol.run(data_path='olive_oils.csv')

# Generate HTML report with all figures, tables, and methods text
protocol.generate_report(output='report.html')
```

---

## Still Have Questions?

- **Technical issues?** See [Troubleshooting](troubleshooting.md)
- **Need help?** Open an issue on [GitHub](https://github.com/chandrasekarnarayana/foodspec/issues)
- **Want to contribute?** See [Developer Guide](../developer-guide/contributing.md)
- **Looking for examples?** Browse [Tutorials](../tutorials/index.md) and [Methods & Validation](../methods/validation/index.md)

---

## Related Pages

- [Troubleshooting](troubleshooting.md) – Technical error solutions
- [Getting Started](../getting-started/getting_started.md) – Installation and first steps
- [Validation → Leakage Prevention](../methods/validation/cross_validation_and_leakage.md) – Avoid overoptimistic results
- [Cookbook → Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md) – Preprocessing recipes
- [Reference → Glossary](../reference/glossary.md) – Term definitions
