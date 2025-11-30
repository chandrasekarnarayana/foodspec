"""Synthetic hyperspectral demo for foodspec."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet
from foodspec.core.hyperspectral import HyperSpectralCube
from foodspec.viz.hyperspectral import plot_hyperspectral_intensity_map


def main() -> None:
    height, width = 5, 5
    wn = np.linspace(1500, 1800, 50)

    # Spectral peak around 1655 cm^-1
    spectral_peak = np.exp(-0.5 * ((wn - 1655) / 5) ** 2)

    # Spatial 2D Gaussian pattern
    xs, ys = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    spatial = np.exp(-((xs - 2) ** 2 + (ys - 2) ** 2) / (2 * 1.0**2))

    rng = np.random.default_rng(0)
    cube = np.zeros((height, width, wn.size))
    for i in range(height):
        for j in range(width):
            cube[i, j, :] = spatial[i, j] * spectral_peak + rng.normal(0, 0.01, size=wn.size)

    meta = pd.DataFrame({"sample_id": [f"p{k}" for k in range(height * width)]})
    hyper = HyperSpectralCube(cube=cube, wavenumbers=wn, metadata=meta, image_shape=(height, width))

    # Round-trip through FoodSpectrumSet
    spectra_flat = hyper.to_spectrum_set(modality="raman")
    hyper_roundtrip = HyperSpectralCube.from_spectrum_set(spectra_flat, image_shape=(height, width))

    print(f"Cube shape: {hyper.cube.shape}, round-trip cube shape: {hyper_roundtrip.cube.shape}")
    print(f"Wavenumbers length: {hyper.wavenumbers.size}")

    # Plot intensity map around 1655 cm^-1
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_hyperspectral_intensity_map(hyper, target_wavenumber=1655, window=5, ax=ax)
    fig.tight_layout()
    plt.savefig("hyperspectral_intensity.png", dpi=150)
    plt.close(fig)

    print("Saved hyperspectral_intensity.png")


if __name__ == "__main__":
    main()
