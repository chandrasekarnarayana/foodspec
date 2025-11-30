"""Hyperspectral visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from foodspec.core.hyperspectral import HyperSpectralCube

__all__ = ["plot_hyperspectral_intensity_map", "plot_ratio_map", "plot_cluster_map"]


def plot_hyperspectral_intensity_map(
    cube: HyperSpectralCube,
    target_wavenumber: float,
    window: float = 5.0,
    ax=None,
):
    """Plot intensity map integrated around a target wavenumber."""
    ax = ax or plt.gca()
    wn = cube.wavenumbers
    mask = (wn >= target_wavenumber - window) & (wn <= target_wavenumber + window)
    if not np.any(mask):
        raise ValueError("No wavenumbers within specified window.")
    integrated = cube.cube[:, :, mask].sum(axis=2)
    im = ax.imshow(integrated, origin="lower")
    ax.set_title(f"Intensity map @ {target_wavenumber}±{window} cm$^{{-1}}$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_ratio_map(cube: HyperSpectralCube, num1: float, num2: float, ax=None, window: float = 3.0):
    """Plot a ratio map of intensities at two wavenumbers."""
    ax = ax or plt.gca()
    wn = cube.wavenumbers
    mask1 = (wn >= num1 - window) & (wn <= num1 + window)
    mask2 = (wn >= num2 - window) & (wn <= num2 + window)
    if not np.any(mask1) or not np.any(mask2):
        raise ValueError("No wavenumbers within specified windows for ratio.")
    int1 = cube.cube[:, :, mask1].mean(axis=2)
    int2 = cube.cube[:, :, mask2].mean(axis=2)
    denom = np.where(int2 == 0, np.finfo(float).eps, int2)
    ratio = int1 / denom
    im = ax.imshow(ratio, origin="lower")
    ax.set_title(f"Ratio map {num1}/{num2} cm$^{{-1}}$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_cluster_map(label_image: np.ndarray, ax=None):
    """Plot a label/cluster map as an image with a colorbar."""
    ax = ax or plt.gca()
    im = ax.imshow(label_image, origin="lower", cmap="tab20")
    ax.set_title("Cluster / segmentation map")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
