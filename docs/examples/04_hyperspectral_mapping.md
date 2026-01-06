# Hyperspectral Mapping: Spatial Analysis

**Level**: Intermediate → Advanced  
**Runtime**: ~3 seconds  
**Key Concepts**: 3D data, segmentation, ROI extraction, spatial analysis

---

## What You Will Learn

In this example, you'll learn how to:
- Load and process 3D hyperspectral data cubes
- Segment images to identify regions of interest (ROIs)
- Extract per-ROI mean spectra
- Perform spectral analysis on aggregated regions
- Visualize spatial patterns and quality variations

After completing this example, you'll understand how to detect defects, map contaminants, track ripeness, and assess spatial quality in food products.

---

## Prerequisites

- Understanding of 2D images and 3D data structures
- Familiarity with clustering and segmentation concepts
- `numpy`, `matplotlib`, `scipy` installed
- Basic knowledge of spectral data (optional, we explain)

**Optional background**: Read [Hyperspectral Mapping Workflow](../workflows/spatial/hyperspectral_mapping.md)

---

## The Problem

**Real-world scenario**: You've captured a hyperspectral image of an apple's surface (width × height × wavelengths). The apple has a bruise you want to detect. Can you:
1. Load the 3D hyperspectral cube?
2. Segment the image to find bruised areas?
3. Extract spectra from the bruised vs. healthy regions?
4. Quantify the quality difference?

**Data**: Synthetic 3D cube (50 × 50 pixels, 1500 wavelengths per pixel)

**Goal**: Segment → Extract → Analyze → Visualize

---

## Step 1: Create & Load Hyperspectral Data

```python
import numpy as np
from pathlib import Path

# Create synthetic 3D hyperspectral cube
np.random.seed(42)
height, width = 50, 50
n_wavelengths = 1500
wavelengths = np.linspace(800, 3000, n_wavelengths)

# Initialize cube with healthy apple spectrum
healthy_spectrum = np.exp(-((wavelengths - 1600) / 200) ** 2)
hsi_cube = np.tile(healthy_spectrum, (height, width, 1))

# Add bruised region (lower intensity at some wavelengths)
y_bruise, x_bruise = slice(20, 35), slice(15, 30)
bruise_spectrum = healthy_spectrum * 0.6  # degraded
hsi_cube[y_bruise, x_bruise, :] = bruise_spectrum

# Add noise
hsi_cube += 0.02 * np.random.randn(height, width, n_wavelengths)
hsi_cube = np.clip(hsi_cube, 0, None)  # ensure non-negative

print(f"Hyperspectral cube shape: {hsi_cube.shape}")
print(f"  Height: {height} pixels")
print(f"  Width: {width} pixels")
print(f"  Wavelengths: {n_wavelengths}")
print(f"Data range: {hsi_cube.min():.3f} to {hsi_cube.max():.3f}")
```

**What's happening**:
- `hsi_cube[y, x, λ]`: Intensity at position (y, x) and wavelength λ
- Healthy region: High intensity
- Bruised region: Lower intensity (quality indicator)
- Realistic noise added throughout

---

## Step 2: Segment the Image

```python
from scipy.ndimage import label
from sklearn.cluster import KMeans

# Simple segmentation: K-means on mean spectrum per pixel
mean_spectrum = hsi_cube.mean(axis=2)  # average spectrum at each pixel

# Reshape for clustering
mean_reshaped = mean_spectrum.reshape(-1, 1)

# K-means clustering (healthy vs. bruised)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_flat = kmeans.fit_predict(mean_reshaped)
labels = labels_flat.reshape(height, width)

# Ensure label 1 is bruised (lower mean intensity)
if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
    labels = 1 - labels

print(f"Segmentation completed: {np.sum(labels == 0)} healthy, {np.sum(labels == 1)} bruised pixels")
```

**What's happening**:
- K-means clusters pixels based on their mean spectral intensity
- This separates healthy (high intensity) from bruised (low intensity) regions
- Labels: 0 = healthy, 1 = bruised

---

## Step 3: Extract Per-ROI Spectra

```python
# Extract mean spectrum for each region
roi_healthy = hsi_cube[labels == 0, :].mean(axis=0)
roi_bruised = hsi_cube[labels == 1, :].mean(axis=0)

print(f"Healthy ROI spectrum shape: {roi_healthy.shape}")
print(f"Bruised ROI spectrum shape: {roi_bruised.shape}")

# Calculate quality metric (e.g., peak intensity ratio)
healthy_peak = roi_healthy.max()
bruised_peak = roi_bruised.max()
quality_ratio = bruised_peak / healthy_peak

print(f"\nQuality Assessment:")
print(f"  Healthy peak intensity: {healthy_peak:.4f}")
print(f"  Bruised peak intensity: {bruised_peak:.4f}")
print(f"  Quality ratio (bruised/healthy): {quality_ratio:.3f}")
print(f"  Quality loss: {(1 - quality_ratio)*100:.1f}%")
```

**Interpretation**:
- Healthy region has higher spectral intensity (good quality)
- Bruised region has lower intensity (degradation)
- Quality ratio < 1 indicates defect

---

## Step 4: Visualize Results

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)

# Plot 1: Segmentation map
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(labels, cmap="RdYlGn_r", vmin=0, vmax=1)
ax1.set_title("Segmentation: Healthy (Green) vs. Bruised (Red)")
ax1.set_xlabel("X (pixels)")
ax1.set_ylabel("Y (pixels)")
plt.colorbar(im1, ax=ax1, label="Label (0=Healthy, 1=Bruised)")

# Plot 2: Mean spectrum per pixel
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(mean_spectrum, cmap="viridis")
ax2.set_title("Mean Spectral Intensity (Quality Map)")
ax2.set_xlabel("X (pixels)")
ax2.set_ylabel("Y (pixels)")
plt.colorbar(im2, ax=ax2, label="Mean Intensity")

# Plot 3: ROI spectra comparison
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(wavelengths, roi_healthy, label="Healthy ROI", linewidth=2, color="green")
ax3.plot(wavelengths, roi_bruised, label="Bruised ROI", linewidth=2, color="red")
ax3.fill_between(wavelengths, roi_healthy, alpha=0.2, color="green")
ax3.fill_between(wavelengths, roi_bruised, alpha=0.2, color="red")
ax3.set_xlabel("Wavenumber (cm⁻¹)")
ax3.set_ylabel("Intensity")
ax3.set_title("Region-of-Interest Spectra")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Difference spectrum
ax4 = fig.add_subplot(gs[1, 1])
difference = roi_healthy - roi_bruised
ax4.plot(wavelengths, difference, linewidth=2, color="purple")
ax4.fill_between(wavelengths, difference, alpha=0.3, color="purple")
ax4.axhline(0, color="black", linestyle="--", alpha=0.5)
ax4.set_xlabel("Wavenumber (cm⁻¹)")
ax4.set_ylabel("Intensity Difference")
ax4.set_title("Healthy - Bruised (Negative = Quality Loss)")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hyperspectral_segmentation.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Figure interpretation**:
- **Top-left**: Spatial map shows where defects are located
- **Top-right**: Mean intensity image (lighter = healthier)
- **Bottom-left**: Extracted spectra (red = degraded, green = healthy)
- **Bottom-right**: Difference spectrum highlights degradation features

---

## Full Working Script

See the production script with complete hyperspectral dataset handling:

📄 **[`examples/hyperspectral_demo.py`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/hyperspectral_demo.py)** – Full working code (114 lines)

---

## Generated Figure

![Hyperspectral Segmentation](https://github.com/chandrasekarnarayana/foodspec/raw/mahttps://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/hyperspectral_demo/hsi_label_map.png)

---

## Key Takeaways

✅ **3D data handling**: Load → Reshape → Segment → Extract ROIs  
✅ **Segmentation**: K-means clusters similar pixels (quality-based)  
✅ **ROI analysis**: Extract mean spectra from spatial regions  
✅ **Quality mapping**: Visualize spatial variations in food quality  

---

## Real-World Applications

- 🍎 **Defect detection**: Locate bruises, dark spots, blemishes on produce
- 🌶️ **Color consistency**: Map ripeness or disease on fruit surfaces
- 🥕 **Contamination tracking**: Identify disease, mold, or foreign material
- 🥗 **Leaf quality**: Assess lettuce or spinach freshness across bundles
- 🥛 **Surface inspection**: Detect cracks, discoloration on dairy products

---

## Advanced Topics

**Want to go deeper?**
- **Spectral unmixing**: Decompose multi-component regions
- **Temporal tracking**: Monitor ROIs over time (ripening, spoilage)
- **3D reconstructions**: Build realistic models of product quality
- **Machine learning**: Use deep learning for pixel-wise classification

See [Hyperspectral Mapping Workflow](../workflows/spatial/hyperspectral_mapping.md) for complete details.

---

## Next Steps

1. **Try it**: Use your own hyperspectral images of food products
2. **Explore**: Change number of clusters (n_clusters=3, 4, etc.)
3. **Learn more**: Read [Hyperspectral Mapping](../workflows/spatial/hyperspectral_mapping.md)
4. **Combine**: Add [Heating Quality](02_heating_quality_monitoring.md) for temporal + spatial analysis

---

## Interactive Notebook

For step-by-step exploration with parameter variations:

📓 **[`examples/tutorials/04_hyperspectral_mapping_teaching.ipynb`](https://github.com/chandrasekarnarayana/foodspec/blob/mahttps://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/04_hyperspectral_mapping_teaching.ipynb)**

---

## Figure provenance
- Generated by [scripts/generate_docs_figures.py](https://github.com/chandrasekarnarayana/foodspec/blob/main/scripts/generate_docs_figures.py)
- Outputs: [../assets/figures/hsi_label_map.png](../assets/figures/hsi_label_map.png) and [../assets/figures/roi_spectra.png](../assets/figures/roi_spectra.png)

