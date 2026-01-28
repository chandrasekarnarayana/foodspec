"""Spectra visualization helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from foodspec.data_objects.spectra_set import FoodSpectrumSet

__all__ = ["plot_spectra", "plot_mean_spectrum"]


def plot_spectra(
    spectra: FoodSpectrumSet,
    sample_indices: Optional[Sequence[int]] = None,
    color_by: Optional[str] = None,
    ax=None,
):
    """Plot spectra lines optionally colored by metadata column."""

    ax = ax or plt.gca()
    X = spectra.x
    wn = spectra.wavenumbers
    meta = spectra.metadata

    if sample_indices is None:
        sample_indices = range(len(spectra))

    colors = None
    if color_by is not None:
        if color_by not in meta.columns:
            raise ValueError(f"Metadata column '{color_by}' not found.")
        colors = pd.Categorical(meta.loc[sample_indices, color_by])
        color_map = plt.get_cmap("tab10")
        color_lookup = {cat: color_map(i % 10) for i, cat in enumerate(colors.categories)}

    for idx in sample_indices:
        color = None
        label = None
        if colors is not None:
            label = meta.loc[idx, color_by]
            color = color_lookup[label]
        ax.plot(wn, X[idx, :], label=label, color=color, alpha=0.8)

    if colors is not None:
        handles, labels = ax.get_legend_handles_labels()
        # deduplicate legend entries
        seen = {}
        new_handles = []
        new_labels = []
        for h, lbl in zip(handles, labels):
            if lbl not in seen:
                seen[lbl] = True
                new_handles.append(h)
                new_labels.append(lbl)
        ax.legend(new_handles, new_labels, title=color_by)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.invert_xaxis()  # common convention for spectra
    return ax


def plot_mean_spectrum(
    spectra: FoodSpectrumSet,
    group_by: Optional[str] = None,
    ax=None,
):
    """Plot mean ± std of spectra, optionally grouped by metadata column."""

    ax = ax or plt.gca()
    wn = spectra.wavenumbers
    meta = spectra.metadata
    X = spectra.x

    if group_by is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        ax.plot(wn, mean, color="C0", label="mean")
        ax.fill_between(wn, mean - std, mean + std, color="C0", alpha=0.2, label="±1 std")
    else:
        if group_by not in meta.columns:
            raise ValueError(f"Metadata column '{group_by}' not found.")
        for i, (group, idxs) in enumerate(meta.groupby(group_by).groups.items()):
            Xg = X[list(idxs)]
            mean = Xg.mean(axis=0)
            std = Xg.std(axis=0)
            color = f"C{i}"
            ax.plot(wn, mean, color=color, label=str(group))
            ax.fill_between(wn, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.invert_xaxis()
    ax.legend()
    return ax
