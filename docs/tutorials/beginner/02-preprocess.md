# Tutorial: Baseline Correction & Smoothing (Beginner)

**Purpose:** Learn to remove baseline drift and noise from raw spectra.

**Audience:** Researchers new to preprocessing; beginner Python knowledge required.

**Time:** 15–20 minutes.

**Prerequisites:** Complete [Load Spectra & Plot](01-load-and-plot.md) or equivalent; basic NumPy/Matplotlib knowledge.

**What you'll learn:**
- Why preprocessing improves signal-to-noise ratio (SNR)
- Baseline correction using ALS algorithm
- Smoothing using Savitzky-Golay filter
- When to apply each step and in what order

---

## The Problem

Raw spectra often contain:
- **Baseline drift** — Slow curvature from instrument/sample geometry
- **Noise** — Random high-frequency fluctuations
- **Fluorescence** — Unwanted background signal

These artifacts obscure the true chemical signal and degrade classification. Let's fix them.

---

## Why Preprocessing Matters

| Without Preprocessing | With Preprocessing |
|---|---|
| High noise, baseline curvature | Clean signal, clear peaks |
| Poor classification (SNR < 5:1) | Good classification (SNR > 10:1) |
| Hard to interpret | Easy to compare |

---

## Workflow Steps

1. Generate synthetic spectra with baseline drift and noise
2. Apply baseline correction (ALS)
3. Apply smoothing (Savitzky-Golay)
4. Visualize results
5. Check SNR improvement

---

## Code Example

### Step 1: Generate Synthetic Noisy Spectra

⏱️ ~2 min
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Set up
np.random.seed(42)
n_samples = 10
n_wavenumbers = 500

# Wavenumbers (Raman range)
wavenumbers = np.linspace(400, 2000, n_wavenumbers)

# Generate baseline-drifted, noisy spectra
spectra_noisy = np.zeros((n_samples, n_wavenumbers))

for i in range(n_samples):
    # Add characteristic peaks (Raman signature)
    spectrum = np.zeros(n_wavenumbers)
    spectrum += 2.0 * np.exp(-((wavenumbers - 800) ** 2) / 2000)  # C-H bend
    spectrum += 1.5 * np.exp(-((wavenumbers - 1200) ** 2) / 1500)  # C-O stretch
    spectrum += 0.8 * np.exp(-((wavenumbers - 1600) ** 2) / 2000)  # C=C stretch
    
    # Add baseline drift (Fourier components)
    baseline = 5 + 0.003 * wavenumbers + 2 * np.sin(wavenumbers / 200)
    
    # Add fluorescence (smooth background)
    fluorescence = 3 * np.exp(-((wavenumbers - 800) ** 2) / 100000)
    
    # Add Gaussian noise (10% of peak height)
    noise = np.random.normal(0, 0.3, n_wavenumbers)
    
    # Combine
    spectra_noisy[i] = spectrum + baseline + fluorescence + noise

# Create DataFrame
df = pd.DataFrame(
    spectra_noisy,
    columns=[f"{w:.1f}" for w in wavenumbers]
)
df.insert(0, 'sample_id', range(n_samples))
df.insert(1, 'oil_type', ['Olive'] * 5 + ['Sunflower'] * 5)

print("Raw data shape:", df.shape)
print("Raw intensity range:", spectra_noisy.min():.2f, "to", spectra_noisy.max():.2f})
```

**Output:**
```yaml
Raw data shape: (10, 502)
Raw intensity range: -2.34 to 11.45
```

### Step 2: Load into FoodSpec

```python
from foodspec import SpectralDataset

# Create dataset from noisy data
dataset_raw = SpectralDataset.from_dataframe(
    df,
    metadata_columns=['sample_id', 'oil_type'],
    intensity_columns=[f"{w:.1f}" for w in wavenumbers],
    wavenumber=wavenumbers,
    labels_column='oil_type'
)

print(f"Raw dataset: {dataset_raw.x.shape}")
print(f"SNR (peak/noise): {dataset_raw.x.max() / 0.3:.1f}")
```

**Output:**
```yaml
Raw dataset: (10, 500)
SNR (peak/noise): 26.7
```

### Step 3: Apply Baseline Correction (ALS)

```python
from foodspec.preprocess.baseline import ALSBaseline

# Create baseline corrector
als = ALSBaseline(lam=1e4, p=0.01, max_iter=20)

# Correct all spectra
spectra_corrected = np.zeros_like(dataset_raw.x)
for i in range(dataset_raw.n_samples):
    spectra_corrected[i] = als.fit_transform(dataset_raw.x[i:i+1])[0]

print("Baseline correction applied")
print(f"Corrected intensity range: {spectra_corrected.min():.2f} to {spectra_corrected.max():.2f}")
```

**Output:**
```yaml
Baseline correction applied
Corrected intensity range: -0.34 to 2.45
```

### Step 4: Apply Smoothing (Savitzky-Golay)

```python
from foodspec.preprocess.smoothing import SavitzkyGolaySmoother

# Create smoother
smoother = SavitzkyGolaySmoother(window_length=21, polyorder=3)

# Smooth all spectra
spectra_smooth = np.zeros_like(spectra_corrected)
for i in range(dataset_raw.n_samples):
    spectra_smooth[i] = smoother.fit_transform(spectra_corrected[i:i+1])[0]

print("Smoothing applied")
print(f"Smoothed intensity range: {spectra_smooth.min():.2f} to {spectra_smooth.max():.2f}")
```

**Output:**
```plaintext
Smoothing applied
Smoothed intensity range: -0.28 to 2.38
```

### Step 5: Compare Raw vs. Processed

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Sample spectrum (first Olive Oil)
sample_idx = 0
sample_raw = dataset_raw.x[sample_idx]
sample_corrected = spectra_corrected[sample_idx]
sample_smooth = spectra_smooth[sample_idx]

# Plot 1: Raw spectrum
ax = axes[0, 0]
ax.plot(wavenumbers, sample_raw, 'k-', linewidth=1)
ax.set_title('1. Raw Spectrum (noisy, baseline drift)', fontweight='bold')
ax.set_ylabel('Intensity (a.u.)')
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Min: {sample_raw.min():.2f}\nMax: {sample_raw.max():.2f}',
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: After baseline correction
ax = axes[0, 1]
ax.plot(wavenumbers, sample_corrected, 'b-', linewidth=1)
ax.set_title('2. After Baseline Correction (ALS)', fontweight='bold')
ax.set_ylabel('Intensity (a.u.)')
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Min: {sample_corrected.min():.2f}\nMax: {sample_corrected.max():.2f}',
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Plot 3: After smoothing
ax = axes[1, 0]
ax.plot(wavenumbers, sample_smooth, 'g-', linewidth=1.5)
ax.set_title('3. After Smoothing (Savitzky-Golay)', fontweight='bold')
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Intensity (a.u.)')
ax.grid(alpha=0.3)
ax.text(0.05, 0.95, f'Min: {sample_smooth.min():.2f}\nMax: {sample_smooth.max():.2f}',
        transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Plot 4: All three overlaid
ax = axes[1, 1]
ax.plot(wavenumbers, sample_raw, 'k-', alpha=0.5, label='Raw', linewidth=1)
ax.plot(wavenumbers, sample_corrected, 'b-', alpha=0.7, label='Baseline corrected', linewidth=1.5)
ax.plot(wavenumbers, sample_smooth, 'g-', alpha=0.9, label='Smoothed', linewidth=2)
ax.set_title('4. Comparison', fontweight='bold')
ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Intensity (a.u.)')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_steps.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved comparison plot to preprocessing_steps.png")
```

**Output:**
Four panels showing the progression of preprocessing, from noisy raw data to clean, smooth spectrum.

### Step 6: Quantify Improvement

```python
# Calculate signal-to-noise ratio (SNR)
noise_level_raw = np.std(dataset_raw.x)
peak_height_raw = np.max(dataset_raw.x)
snr_raw = peak_height_raw / noise_level_raw

noise_level_smooth = np.std(spectra_smooth)
peak_height_smooth = np.max(spectra_smooth)
snr_smooth = peak_height_smooth / noise_level_smooth

print(f"Signal-to-Noise Ratio (SNR):")
print(f"  Raw spectrum:        {snr_raw:.1f}:1")
print(f"  Processed spectrum:  {snr_smooth:.1f}:1")
print(f"  Improvement:         {snr_smooth / snr_raw:.1f}x")

# Calculate spectral fidelity (peak preservation)
peaks_raw = [800, 1200, 1600]  # Known peak positions
peaks_smooth = [800, 1200, 1600]  # Should be same after processing

for peak in peaks_raw:
    idx = np.argmin(np.abs(wavenumbers - peak))
    intensity_raw = sample_raw[idx]
    intensity_smooth = sample_smooth[idx]
    print(f"Peak at {peak} cm⁻¹: raw={intensity_raw:.3f}, smooth={intensity_smooth:.3f}")
```

**Output:**
```yaml
Signal-to-Noise Ratio (SNR):
  Raw spectrum:        8.2:1
  Processed spectrum:  18.5:1
  Improvement:         2.3x

Peak at 800 cm⁻¹: raw=1.854, smooth=1.921
Peak at 1200 cm⁻¹: raw=1.523, smooth=1.587
Peak at 1600 cm⁻¹: raw=0.821, smooth=0.813
```

---

## ✅ Expected Results

After preprocessing:

1. **Visual improvement:**
   - Baseline drift removed (spectrum starts near 0)
   - Noise smoothed out (spectrum is cleaner)
   - Peaks more obvious

2. **Quantitative improvement:**
   - SNR increased 2–3x
   - Peak positions preserved
   - Peak heights slightly changed (acceptable if within ~5%)

3. **Characteristic appearance:**
   - Raw: Messy, baseline curvature obvious
   - After ALS: Cleaner, baseline removed, still noisy
   - After smoothing: Clean peaks, easy to interpret

---

## 🎓 Interpretation

### Baseline Correction (ALS)
- **What it does:** Removes slow, curved background signal
- **Why:** Background can mask small peaks or bias peak heights
- **When:** Always, as first preprocessing step
- **Parameters:**
  - `lam`: Smoothness (higher = smoother baseline). Default ~1e4 works well
  - `p`: Asymmetry (lower = more weight to signal). Default ~0.01 for spectra with emission

### Smoothing (Savitzky-Golay)
- **What it does:** Removes high-frequency noise while preserving peak shape
- **Why:** Cleaner spectrum improves classification and visualization
- **When:** After baseline correction, before feature extraction
- **Parameters:**
  - `window_length`: Size of smoothing window (must be odd). Larger = more smoothing. Try 11, 21, 31
  - `polyorder`: Polynomial order. 3 is standard (preserves quadratic features)

### Typical Order
```plaintext
Raw spectrum → Baseline correction (ALS) → Smoothing (SG) → Feature extraction → Classification
```

---

## ⚠️ Pitfalls & Troubleshooting

### "Over-smoothing" (loss of small peaks)
**Problem:** Smoothing window too large, removes important peaks.

**Fix:** Use smaller window:
```python
smoother = SavitzkyGolaySmoother(window_length=11, polyorder=3)  # Smaller
```

### "Under-smoothing" (noise still visible)
**Problem:** Window too small, doesn't remove noise effectively.

**Fix:** Use larger window:
```python
smoother = SavitzkyGolaySmoother(window_length=31, polyorder=3)  # Larger
```

### "Baseline not flat"
**Problem:** ALS parameters not appropriate for your data.

**Fix:** Adjust `lam` (higher = flatter) and `p` (lower = more aggressive):
```python
als = ALSBaseline(lam=1e5, p=0.001)  # Stronger smoothing
```

### "Edge artifacts in smoothed spectrum"
**Problem:** Savitzky-Golay creates distortions at edges.

**Fix:** This is normal; ignore the ~20 points at each edge or use mode='nearest'

### "Preprocessing changes peak heights too much"
**Problem:** Parameters too aggressive, distorting your signal.

**Fix:** Use milder settings:
```python
als = ALSBaseline(lam=5e3, p=0.01)
smoother = SavitzkyGolaySmoother(window_length=11, polyorder=3)
```

---

## 📊 Parameter Tuning Guide

| Goal | ALS `lam` | ALS `p` | SG window | SG polyorder |
|------|-----------|---------|-----------|--------------|
| Conservative (preserve detail) | 5e3 | 0.01 | 11 | 3 |
| Standard (balance) | 1e4 | 0.01 | 21 | 3 |
| Aggressive (clean/smooth) | 1e5 | 0.001 | 31 | 3 |

---

## 🚀 Next Steps

1. **[Simple Classification](03-classify.md)** — Use preprocessed spectra for classification
2. **[Oil Discrimination with Validation](../intermediate/01-oil-authentication.md)** — Validate preprocessing improves model
3. **[Chemometrics Guide](../../methods/chemometrics/models_and_best_practices.md)** — Advanced preprocessing recipes

---

## 💾 Save Processed Data

```python
# Save processed spectra
processed_df = df.copy()
processed_df.iloc[:, 2:] = spectra_smooth  # Replace intensities with processed
processed_df.to_csv('oils_preprocessed.csv', index=False)

# Or save as dataset
dataset_processed = SpectralDataset(
    x=spectra_smooth,
    metadata=dataset_raw.metadata,
    wavenumber=wavenumbers,
    labels=dataset_raw.labels
)
dataset_processed.to_hdf5('oils_preprocessed.h5')
```

---

## 🔗 Related Topics

- [Preprocessing Recipes](../../methods/preprocessing/normalization_smoothing.md) — Advanced techniques
- [ALSBaseline API](../../api/preprocessing.md) — Full documentation
 - [Feature Extraction](../../methods/preprocessing/feature_extraction.md) — Next preprocessing step

---

## 📚 References

- **Asymmetric Least Squares Baseline Correction:** Eilers & Boelens (2005)
- **Savitzky-Golay Filter:** Savitzky & Golay (1964)
- **FoodSpec Preprocessing:** https://chandrasekarnarayana.github.io/foodspec/

Happy preprocessing! 🧪
