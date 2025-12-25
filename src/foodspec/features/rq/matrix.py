"""
Oil vs chips matrix divergence analysis for RQ.

Provides `compare_oil_vs_chips()` and helper logic to compute divergence
between matrix types on ratio/feature columns.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .types import RQConfig
from .utils import _cv


def _trend(df: pd.DataFrame, heating_col: str, feat: str) -> tuple[float, float]:
    """Compute linear trend slope and p-value for `feat` vs `heating_col`."""
    if heating_col not in df.columns or df.empty:
        return (np.nan, np.nan)
    x = pd.to_numeric(df[heating_col], errors="coerce")
    y = df[feat]
    mask = ~x.isna() & ~y.isna()
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return (np.nan, np.nan)
    slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
    return (float(slope), float(p_val))


def compare_oil_vs_chips(df: pd.DataFrame, config: RQConfig, features: Sequence[str]) -> pd.DataFrame:
    """
    Compare stability and heating trends between matrix types (oil vs chips).

    Flags divergence when slopes differ in significance/sign or means differ.
    """
    matrix_col = config.matrix_col
    heating_col = config.heating_col

    if matrix_col not in df.columns:
        return pd.DataFrame(columns=["feature", "oil_cv", "chips_cv", "oil_slope", "chips_slope", "diverges"])

    rows = []
    for feat in features:
        oil_sub = df[df[matrix_col] == "oil"]
        chips_sub = df[df[matrix_col] == "chips"]

        oil_cv = _cv(oil_sub[feat])["cv_percent"] if not oil_sub.empty else np.nan
        chips_cv = _cv(chips_sub[feat])["cv_percent"] if not chips_sub.empty else np.nan

        oil_slope, oil_p = _trend(oil_sub, heating_col, feat)
        chips_slope, chips_p = _trend(chips_sub, heating_col, feat)

        # Mean diff and Cohen's d
        oil_vals = oil_sub[feat].dropna().astype(float)
        chips_vals = chips_sub[feat].dropna().astype(float)
        mean_diff = oil_vals.mean() - chips_vals.mean() if len(oil_vals) and len(chips_vals) else np.nan
        cohen_d = np.nan
        if len(oil_vals) > 1 and len(chips_vals) > 1:
            pooled_std = np.sqrt(((oil_vals.std(ddof=1) ** 2) + (chips_vals.std(ddof=1) ** 2)) / 2)
            if pooled_std != 0:
                cohen_d = mean_diff / pooled_std
        try:
            t_stat, p_mean = stats.ttest_ind(oil_vals, chips_vals, equal_var=False, nan_policy="omit")
        except Exception:
            p_mean = np.nan

        # Spearman trends
        def _spearman(sub: pd.DataFrame) -> tuple[float, float]:
            if heating_col not in sub.columns:
                return (np.nan, np.nan)
            x = pd.to_numeric(sub[heating_col], errors="coerce")
            y = sub[feat].astype(float)
            mask = ~x.isna() & ~y.isna()
            if mask.sum() < 3:
                return (np.nan, np.nan)
            rho, pval = stats.spearmanr(x[mask], y[mask])
            return (float(rho), float(pval))

        oil_rho, oil_rho_p = _spearman(oil_sub)
        chips_rho, chips_rho_p = _spearman(chips_sub)

        # Divergence criteria
        diverges = False
        if np.isfinite(oil_p) and np.isfinite(chips_p):
            sig_oil = oil_p < 0.05
            sig_chips = chips_p < 0.05
            if sig_oil != sig_chips:
                diverges = True
            elif sig_oil and sig_chips and np.sign(oil_slope) != np.sign(chips_slope):
                diverges = True
        if np.isfinite(p_mean) and p_mean < 0.05:
            diverges = True

        rows.append(
            {
                "feature": feat,
                "oil_cv": oil_cv,
                "chips_cv": chips_cv,
                "oil_slope": oil_slope,
                "chips_slope": chips_slope,
                "oil_p_value": oil_p,
                "chips_p_value": chips_p,
                "diverges": bool(diverges),
                "mean_diff": mean_diff,
                "p_mean": p_mean,
                "cohen_d": cohen_d,
                "delta_cv": oil_cv - chips_cv if np.isfinite(oil_cv) and np.isfinite(chips_cv) else np.nan,
                "delta_slope": (
                    oil_slope - chips_slope if np.isfinite(oil_slope) and np.isfinite(chips_slope) else np.nan
                ),
                "oil_spearman_rho": oil_rho,
                "chips_spearman_rho": chips_rho,
                "oil_spearman_p": oil_rho_p,
                "chips_spearman_p": chips_rho_p,
            }
        )

    df_out = pd.DataFrame(rows)
    if config.adjust_p_values and not df_out.empty:
        # Adjust mean p-values
        if "p_mean" in df_out.columns:
            reject, p_adj, _, _ = multipletests(df_out["p_mean"].fillna(1.0), method="fdr_bh")
            df_out["p_mean_adj"] = p_adj
            df_out["significant_mean_fdr"] = reject
        # Adjust trend p-values (combine oil/chips p via max conservative)
        trend_p = np.maximum(df_out["oil_p_value"].fillna(1.0), df_out["chips_p_value"].fillna(1.0))
        reject, p_adj, _, _ = multipletests(trend_p, method="fdr_bh")
        df_out["p_trend_adj"] = p_adj
        df_out["significant_trend_fdr"] = reject

    # Interpretation tags
    tags: List[str] = []
    for _, r in df_out.iterrows():
        tag = "matrix-robust marker"
        if r.get("significant_trend_fdr") and np.sign(r["oil_slope"]) != np.sign(r["chips_slope"]):
            tag = "opposite trend in oil vs chips"
        elif r.get("significant_trend_fdr") and r["oil_p_value"] < 0.05 and r["chips_p_value"] >= 0.05:
            tag = "stable in chips, trending in oil"
        elif r.get("significant_trend_fdr") and r["chips_p_value"] < 0.05 and r["oil_p_value"] >= 0.05:
            tag = "stable in oil, trending in chips"
        elif r.get("significant_mean_fdr"):
            tag = "mean shift between matrices"
        tags.append(tag)
    if not df_out.empty:
        df_out["interpretation"] = tags
        df_out["diverges"] = pd.Series([bool(x) for x in df_out["diverges"]], dtype=object)

    return df_out
