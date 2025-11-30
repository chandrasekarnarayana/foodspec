# Hyperspectral Support

`HyperSpectralCube` represents 2D Raman/FTIR maps stored as `(height, width, n_points)`.

## Converting between cube and spectrum set

```python
from foodspec.core.hyperspectral import HyperSpectralCube
from foodspec.core.dataset import FoodSpectrumSet

# From spectrum set (flattened pixels) to cube
cube = HyperSpectralCube.from_spectrum_set(spectra, image_shape=(height, width))

# Back to spectrum set with pixel coords
spectra_flat = cube.to_spectrum_set(modality="raman")
```

Existing metadata is preserved; `row` and `col` are added (existing columns are not overwritten).

## Inspecting spectra

```python
spectrum = cube.get_pixel_spectrum(row=0, col=1)
mean_spec = cube.mean_spectrum()
```

## Plotting intensity map

```python
from foodspec.viz.hyperspectral import plot_hyperspectral_intensity_map
import matplotlib.pyplot as plt

ax = plot_hyperspectral_intensity_map(cube, target_wavenumber=1655, window=5)
plt.show()

## Example script

See `examples/hyperspectral_demo.py` for a small synthetic demo that:

- builds a HyperSpectralCube from synthetic data,
- converts it to and from a FoodSpectrumSet,
- plots an intensity map around 1655 cm^-1 and saves it as `hyperspectral_intensity.png`.
```
