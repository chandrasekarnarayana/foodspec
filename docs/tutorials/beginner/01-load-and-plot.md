# Tutorial: Load Spectra & Plot (Level 1)

**Goal:** Load spectral data from CSV and create your first plots. Understand the fundamental data structure.

**Time:** 5–10 minutes

**Prerequisites:** Python 3.10+, FoodSpec installed (`pip install foodspec`)

**What You'll Learn:**
- Load spectral data into memory
- Understand metadata vs. spectral intensities
- Create basic plots of raw spectra
- Use the `SpectralDataset` class

---

## 🎯 The Problem

Raw spectral data looks like numbers in a spreadsheet. Let's learn how to:
1. Load it into FoodSpec
2. Visualize it
3. Understand what you're looking at

---

## 📊 Data Format

FoodSpec expects CSV data with this structure:

```csv
oil_type,heating_stage,batch,1000.5,1001.2,1002.1,...,4000.0
Olive Oil,0,A,0.45,0.47,0.49,...,0.12
Olive Oil,0,A,0.46,0.48,0.50,...,0.11
Sunflower Oil,0,B,0.52,0.54,0.56,...,0.15
```

**Columns:**
- **Metadata columns** (left): `oil_type`, `heating_stage`, `batch`, etc.
- **Intensity columns** (right): Wavenumbers as column names (e.g., `1000.5`, `1001.2`)

**Requirements:**
- Metadata column names: arbitrary (e.g., `sample_id`, `oil_type`, `heating_stage`)
- Wavenumber columns: must be numeric (float) values representing the spectral axis
- No missing intensity values (NaN); use baseline correction if needed
- Labels column recommended for classification (e.g., `oil_type`, `material`)

---

## 🔨 Steps

1. Generate or load sample data
2. Create a `SpectralDataset` from the CSV
3. Explore the dataset structure
4. Plot raw spectra
5. Plot mean spectrum with error bands

---

## 💻 Code Example

### Step 1: Generate Synthetic Data

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Create synthetic oil spectra (toy data)
np.random.seed(42)
n_samples = 20
n_wavenumbers = 100

# Wavenumbers (simulated Raman range)
wavenumbers = np.linspace(400, 2000, n_wavenumbers)

# Generate synthetic spectra (Gaussian peaks at different positions)
spectra = np.random.randn(n_samples, n_wavenumbers) * 0.1  # noise

# Add characteristic peaks for different oils
for i in range(n_samples):
    if i < 10:  # Olive Oil
        spectra[i] += np.exp(-((wavenumbers - 800) ** 2) / 1000)
        spectra[i] += 0.5 * np.exp(-((wavenumbers - 1200) ** 2) / 1500)
    else:  # Sunflower Oil
        spectra[i] += 1.2 * np.exp(-((wavenumbers - 750) ** 2) / 800)
        spectra[i] += 0.3 * np.exp(-((wavenumbers - 1300) ** 2) / 2000)

# Create DataFrame with metadata
df = pd.DataFrame(
    spectra,
    columns=[f"{w:.1f}" for w in wavenumbers]
)
df.insert(0, 'oil_type', ['Olive Oil'] * 10 + ['Sunflower Oil'] * 10)
df.insert(1, 'batch', ['A'] * 5 + ['B'] * 5 + ['C'] * 5 + ['D'] * 5)

print(df.head())
print(f"Shape: {df.shape}")  # (20, 102) — 20 samples, 2 metadata + 100 wavenumber columns
```

**Output:**
```plaintext
    oil_type batch  400.0   411.1   422.2  ...  1988.9
0  Olive Oil     A   0.05   0.12   0.25  ...   -0.08
1  Olive Oil     A  -0.03   0.10   0.22  ...   -0.10
2  Olive Oil     B   0.08   0.15   0.28  ...   -0.05
...
Shape: (20, 102)
```

### Step 2: Load into FoodSpec

```python
from foodspec import SpectralDataset

# Identify metadata columns (everything before wavenumber columns)
metadata_cols = ['oil_type', 'batch']
intensity_cols = [str(w) for w in wavenumbers]

# Create SpectralDataset
dataset = SpectralDataset.from_dataframe(
    df,
    metadata_columns=metadata_cols,
    intensity_columns=intensity_cols,
    wavenumber=wavenumbers,
    labels_column='oil_type'
)

print(f"Dataset shape: {dataset.x.shape}")  # (20, 100)
print(f"Wavenumber range: {dataset.wavenumber[0]:.1f}–{dataset.wavenumber[-1]:.1f} cm⁻¹")
print(f"Labels: {dataset.labels}")
```

**Output:**
```yaml
Dataset shape: (20, 100)
Wavenumber range: 400.0–2000.0 cm⁻¹
Labels: ['Olive Oil' 'Olive Oil' ... 'Sunflower Oil']
```

### Step 3: Explore the Dataset

```python
# Check structure
print(f"n_samples: {dataset.n_samples}")
print(f"n_features: {dataset.n_features}")
print(f"Unique labels: {np.unique(dataset.labels)}")
print(f"Label counts: {np.unique(dataset.labels, return_counts=True)}")
print(f"Metadata keys: {list(dataset.metadata.keys())}")

# Access specific samples
print(f"\nFirst spectrum shape: {dataset.x[0].shape}")
print(f"First sample metadata: {dataset.metadata['oil_type'][0]}")
```

**Output:**
```yaml
n_samples: 20
n_features: 100
Unique labels: ['Olive Oil' 'Sunflower Oil']
Label counts: (array(['Olive Oil', 'Sunflower Oil'], dtype=object), array([10, 10]))
Metadata keys: ['oil_type', 'batch']

First spectrum shape: (100,)
First sample metadata: Olive Oil
```

### Step 4: Plot Raw Spectra

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: All spectra by class
ax = axes[0]
for label in np.unique(dataset.labels):
    mask = dataset.labels == label
    ax.plot(
        dataset.wavenumber,
        dataset.x[mask].T,
        alpha=0.3,
        label=label
    )
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Intensity (a.u.)')
ax.set_title('Raw Spectra by Oil Type')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Mean spectrum with std bands
ax = axes[1]
for label in np.unique(dataset.labels):
    mask = dataset.labels == label
    mean = dataset.x[mask].mean(axis=0)
    std = dataset.x[mask].std(axis=0)
    ax.plot(dataset.wavenumber, mean, label=label, linewidth=2)
    ax.fill_between(
        dataset.wavenumber,
        mean - std,
        mean + std,
        alpha=0.2
    )
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Mean Intensity (a.u.)')
ax.set_title('Mean Spectrum ± Std Dev')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('raw_spectra.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved plot to raw_spectra.png")
```

**Output:**
Two plots:
- Left: All raw spectra overlaid by oil type (spaghetti plot)
- Right: Mean spectrum for each oil type with uncertainty bands

---

## ✅ Expected Results

After running this tutorial, you should see:

1. **Terminal output** showing:
   - Dataset shape: (20, 100) — 20 samples, 100 wavenumber features
   - Wavenumber range: 400–2000 cm⁻¹
   - Labels: 10 Olive Oil, 10 Sunflower Oil

2. **Plot (`raw_spectra.png`)** showing:
   - Left panel: Individual spectra colored by oil type (messy, overlapping)
   - Right panel: Mean spectra with confidence bands (cleaner, easier to compare)

3. **Observable differences**:
   - Olive Oil peaks at ~800 cm⁻¹ and ~1200 cm⁻¹
   - Sunflower Oil peaks at ~750 cm⁻¹ and ~1300 cm⁻¹
   - Visual separation suggests these oils are distinguishable

---

## 🎓 Interpretation

What does a spectrum tell us?

- **Peak position** (x-axis): Different wavenumbers represent different functional groups
  - ~800 cm⁻¹: C-H bending (structure-dependent)
  - ~1200 cm⁻¹: C-O stretching (saturated vs. unsaturated)
  - ~1600 cm⁻¹: C=C stretching (degree of unsaturation)

- **Peak height** (y-axis): Higher intensity = more of that functional group

- **Overall shape**: Fingerprint unique to each oil type

- **Noise level**: If spectra are very noisy, preprocessing (baseline correction, smoothing) helps

---

## ⚠️ Pitfalls & Troubleshooting

### "ValueError: wavenumber length mismatch"
**Problem:** Wavenumber array length doesn't match number of intensity columns.

**Fix:** Ensure `len(wavenumbers) == len(intensity_cols)`

```python
# Check
print(f"Wavenumbers: {len(wavenumbers)}")
print(f"Intensities: {len(intensity_cols)}")
# Should be equal
```

### "KeyError: 'oil_type' not found"
**Problem:** Metadata column name is misspelled or missing.

**Fix:** Check column names in your DataFrame:

```python
print(df.columns)  # Verify spelling and capitalization
```

### "No visible spectra in plot"
**Problem:** Intensity values are very small (signal buried in noise).

**Fix:** Check intensity value ranges:

```python
print(f"Min intensity: {dataset.x.min()}")
print(f"Max intensity: {dataset.x.max()}")
# If < 0.001, may need baseline correction or check data units
```

### "Plot looks like noise"
**Problem:** Synthetic spectra may lack structure; real data will look better.

**Fix:** Load real data or adjust synthetic peak parameters:

```python
# Make peaks more pronounced
spectra[i] += 2.0 * np.exp(-((wavenumbers - 800) ** 2) / 500)  # Bigger peak
```

---

## 🚀 Next Steps

Once you've mastered loading and plotting:

1. **[Baseline Correction & Smoothing](02-preprocess.md)** — Clean up your spectra
2. **[Simple Classification](03-classify.md)** — Train a model to distinguish oils
3. **[Oil Discrimination with Validation](../intermediate/01-oil-authentication.md)** — Validate your classifier

---

## 💾 Save Your Work

```python
# Save dataset for later use
dataset.to_hdf5('my_oils.h5')

# Or export as CSV
df.to_csv('my_oils.csv', index=False)

# Or as pickle
import pickle
with open('my_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
```

---

## 🔗 Related Topics

- [Data Formats & HDF5](../../user-guide/data_formats_and_hdf5.md) — Deep dive into data structure
- [SpectralDataset API](../../api/core.md) — Complete class documentation
- [Plotting with FoodSpec](../../user-guide/visualization.md) — Advanced visualization

---

## 📚 Resources

- **FoodSpec documentation:** https://chandrasekarnarayana.github.io/foodspec/
- **Example data:** `/examples/data/`
- **Tutorials:** This folder (`02-tutorials/`)

Happy exploring! 🔬
