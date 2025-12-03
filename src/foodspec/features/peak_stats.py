from __future__ import annotations

import pandas as pd


def compute_peak_stats(peaks: pd.DataFrame, metadata: pd.DataFrame | None = None, group_keys=None) -> pd.DataFrame:
    """
    Compute mean/std of peak positions and intensities, optionally grouped.

    Parameters
    ----------
    peaks : pd.DataFrame
        Long-format table with columns: spectrum_id, peak_id (or band_label), position, intensity.
    metadata : pd.DataFrame, optional
        Metadata indexed by spectrum_id (or containing a spectrum_id column) with categorical variables.
    group_keys : list/tuple of str, optional
        Categorical columns in metadata to group by (e.g., ["oil_type"] or ["oil_type", "batch"]).

    Returns
    -------
    pd.DataFrame
        Tidy summary with group columns (if any), peak_id, n_samples, mean/std of position and intensity.
    """

    df = peaks.copy()
    if metadata is not None:
        meta = metadata.copy()
        if "spectrum_id" in meta.columns:
            meta = meta.set_index("spectrum_id")
        df = df.join(meta, on="spectrum_id")

    group_cols = [] if not group_keys else list(group_keys)
    agg_cols = group_cols + ["peak_id"]
    grouped = df.groupby(agg_cols)
    summary = grouped.agg(
        n_samples=("position", "count"),
        mean_pos=("position", "mean"),
        std_pos=("position", "std"),
        mean_intensity=("intensity", "mean"),
        std_intensity=("intensity", "std"),
    ).reset_index()
    return summary


def compute_ratio_table(ratios: pd.DataFrame, metadata: pd.DataFrame | None = None, group_keys=None) -> pd.DataFrame:
    """
    Summarize ratios by group with mean/std and counts.

    Parameters
    ----------
    ratios : pd.DataFrame
        Long or wide table of ratios. Expected columns: spectrum_id, ratio_name, value (long) OR wide with ratio columns.
    metadata : pd.DataFrame, optional
        Metadata indexed by spectrum_id (or with spectrum_id column) for grouping.
    group_keys : list/tuple of str, optional
        Categorical columns in metadata to group by.

    Returns
    -------
    pd.DataFrame
        Tidy table with group columns (if any), ratio_name, n, mean, std.
    """

    df = ratios.copy()
    if "ratio_name" not in df.columns:
        # assume wide; melt to long
        df = df.reset_index(names="spectrum_id") if df.index.name else df
        df = df.melt(id_vars=[c for c in df.columns if c == "spectrum_id"], var_name="ratio_name", value_name="value")
    if metadata is not None:
        meta = metadata.copy()
        if "spectrum_id" in meta.columns:
            meta = meta.set_index("spectrum_id")
        df = df.join(meta, on="spectrum_id")

    group_cols = [] if not group_keys else list(group_keys)
    agg_cols = group_cols + ["ratio_name"]
    grouped = df.groupby(agg_cols)
    summary = grouped.agg(
        n=("value", "count"),
        mean=("value", "mean"),
        std=("value", "std"),
    ).reset_index()
    return summary
