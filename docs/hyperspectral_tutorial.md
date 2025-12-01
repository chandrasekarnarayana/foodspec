# Hyperspectral analysis

## What is a hyperspectral cube?
- A 3D array `(height, width, n_wavenumbers)` where each pixel has a full spectrum.
- FoodSpec wraps this as `HyperSpectralCube`, convertible to/from `FoodSpectrumSet` (flattened pixels with row/col metadata).

## Workflow
1. Load or construct per-pixel spectra (flattened) + image shape.  
2. Build a cube: `cube = HyperSpectralCube.from_spectrum_set(fs_pixels, image_shape=(H, W))`.  
3. Visualization:
   - Intensity map at a target wavenumber: `plot_hyperspectral_intensity_map(cube, target_wavenumber=1655, window=5)`.
   - Ratio map: `plot_ratio_map(cube, num1=1655, num2=1742)`.
   - Cluster map: apply clustering/labels and use `plot_cluster_map`.

## Example (CLI)
```bash
foodspec hyperspectral \
  libraries/hyperspectral_pixels.h5 \
  --height 20 --width 20 \
  --target-wavenumber 1655 \
  --window 5 \
  --output-dir runs/hyper_demo
```
Outputs: intensity_map.png, summary.json, optional mean spectrum CSV.

## Example (Python)
```python
from foodspec.core.hyperspectral import HyperSpectralCube
from foodspec.viz.hyperspectral import plot_hyperspectral_intensity_map, plot_ratio_map
cube = HyperSpectralCube.from_spectrum_set(fs_pixels, image_shape=(20, 20))
plot_hyperspectral_intensity_map(cube, target_wavenumber=1655, window=5)
plot_ratio_map(cube, num1=1655, num2=1742)
```

## Interpretation and reporting
- Intensity/ratio maps show spatial distribution of components or contaminants; hotspots may indicate adulteration or uneven composition.
- Cluster maps summarize segmentation (e.g., phases or contaminants).
- **Main figure**: key intensity/ratio map with color bar and scale.  
- **Supplementary**: representative pixel spectra, mean spectrum, cluster map variations, and any preprocessing notes (cropping/smoothing applied to pixels).

See also
- [metrics_interpretation.md](metrics_interpretation.md)
- [keyword_index.md](keyword_index.md)
- [api_reference.md](api_reference.md)
