#!/usr/bin/env python3
"""Aging workflow quickstart: synthetic storage-time dataset.

Generates a synthetic TimeSpectrumSet with per-entity storage times in days and
an oxidation ratio increasing over time. Fits degradation trajectories and
estimates shelf-life via time-to-threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from foodspec.core.time import TimeSpectrumSet
from foodspec.workflows.aging import compute_degradation_trajectories
from foodspec.workflows.shelf_life import estimate_remaining_shelf_life


def _synthetic_aging_dataset(
    n_entities: int = 5,
    n_timepoints: int = 20,
    time_max_days: float = 180.0,
    replicates_per_time: int = 3,
    random_state: int = 42,
) -> TimeSpectrumSet:
    rng = np.random.default_rng(random_state)
    wavenumbers = np.linspace(600, 1800, 400)

    rows = []
    spectra = []
    for e in range(n_entities):
        entity_id = f"E{e+1}"
        slope = rng.normal(0.003, 0.0007)  # ratio units per day
        intercept = rng.normal(0.35, 0.05)
        times = np.linspace(0.0, time_max_days, n_timepoints)
        for t in times:
            for r in range(replicates_per_time):
                ratio = intercept + slope * t + rng.normal(0.0, 0.03)
                rows.append({
                    "sample_id": entity_id,
                    "days": float(t),
                    "oxidation_ratio": float(ratio),
                })
                # Minimal synthetic spectra (not used by workflow; placeholder)
                base = np.exp(-((wavenumbers - 1655.0) / 200.0) ** 2)
                carbonyl = np.exp(-((wavenumbers - 1742.0) / 150.0) ** 2)
                spectrum = 0.6 * base + (0.2 + ratio * 0.1) * carbonyl + rng.normal(0.0, 0.01, wavenumbers.size)
                spectra.append(spectrum)
    metadata = pd.DataFrame(rows)
    x = np.vstack(spectra).astype(float)
    ds = TimeSpectrumSet(x=x, wavenumbers=wavenumbers, metadata=metadata, modality="ftir", time_col="days", entity_col="sample_id")
    return ds


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Build synthetic storage-time dataset
    ds = _synthetic_aging_dataset()
    print(f"Dataset: {ds.x.shape[0]} spectra, time range {ds.metadata['days'].min():.0f}-{ds.metadata['days'].max():.0f} days")
    print(f"Entities: {ds.metadata['sample_id'].nunique()}")

    # Fit degradation trajectories (linear)
    res = compute_degradation_trajectories(ds, value_col="oxidation_ratio", method="linear")
    print("\nTrajectory metrics (first 5 rows):")
    print(res.metrics.head())

    # Plot trajectories for first two entities
    ents = list(res.fits.keys())[:2]
    fig, ax = plt.subplots(figsize=(8, 5))
    for ent in ents:
        fit = res.fits[ent]
        ax.scatter(fit.times, fit.values, alpha=0.6, label=f"{ent} data")
        ax.plot(fit.times, fit.fitted, label=f"{ent} fit")
    ax.set_xlabel("Storage Time (days)")
    ax.set_ylabel("Oxidation Ratio")
    ax.set_title("Aging: Degradation Trajectories (synthetic)")
    ax.legend()
    plt.tight_layout()
    traj_path = out_dir / "aging_degradation_trajectories.png"
    plt.savefig(traj_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {traj_path}")

    # Shelf-life estimation at threshold
    threshold = 0.9
    sl_df = estimate_remaining_shelf_life(ds, value_col="oxidation_ratio", threshold=threshold)
    print("\nShelf-life estimates (days):")
    print(sl_df.head())

    # Bar chart of t_star with CI
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(sl_df["entity"], sl_df["t_star"], yerr=[sl_df["t_star"] - sl_df["ci_low"], sl_df["ci_high"] - sl_df["t_star"]], capsize=4)
    ax2.set_ylabel("Time to Threshold (days)")
    ax2.set_title(f"Shelf-Life at threshold={threshold}")
    plt.tight_layout()
    shelf_path = out_dir / "aging_shelf_life_estimates.png"
    plt.savefig(shelf_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {shelf_path}")


if __name__ == "__main__":
    main()
