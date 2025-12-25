"""
Text report generation for Ratio-Quality (RQ) analysis.

Provides `generate_text_report()` which formats the RQ results into
human-readable sections for CLI and protocol outputs.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import pandas as pd


def generate_text_report(
    stability: pd.DataFrame,
    discrim: pd.DataFrame,
    heating: pd.DataFrame,
    oil_vs_chips: pd.DataFrame,
    feat_importance: Optional[pd.DataFrame],
    norm_comp: Optional[pd.DataFrame],
    minimal_panel: Optional[pd.DataFrame],
    clustering_metrics: Optional[Mapping[str, Any]],
    warnings: Sequence[str],
    context: Optional[Mapping[str, Any]] = None,
    top_k: int = 5,
) -> str:
    """
    Generate a text report with optional protocol/context fields.

    Parameters
    ----------
    stability : pd.DataFrame
        Stability (CV%) summary.
    discrim : pd.DataFrame
        Discriminative power summary.
    heating : pd.DataFrame
        Heating trend analysis.
    oil_vs_chips : pd.DataFrame
        Matrix divergence analysis.
    feat_importance : Optional[pd.DataFrame]
        Classifier-based feature importance.
    norm_comp : Optional[pd.DataFrame]
        Normalization comparison results.
    minimal_panel : Optional[pd.DataFrame]
        Minimal marker panel selection.
    clustering_metrics : Optional[Dict]
        Clustering structure metrics.
    warnings : List[str]
        Guardrail warnings.
    context : Optional[Dict[str, Any]]
        Additional context like protocol name/version and QC counts.
    top_k : int
        Number of items to show in table sections.

    Returns
    -------
    str
        Rendered report text.
    """
    ctx = context or {}
    tpl: List[str] = []
    tpl.append("Ratio-Quality (RQ) Report")
    tpl.append("=========================")
    if "protocol_name" in ctx:
        tpl.append(f"Protocol: {ctx['protocol_name']}")
    if "protocol_version" in ctx:
        tpl.append(f"Protocol version: {ctx['protocol_version']}")
    if "validation_strategy" in ctx:
        tpl.append(f"Validation strategy: {ctx['validation_strategy']}")
    if "intro" in ctx:
        tpl.append(ctx["intro"])
    if "preprocessing_notes" in ctx:
        tpl.append(f"Preprocessing: {ctx['preprocessing_notes']}")
    if "normalization_modes" in ctx:
        tpl.append(f"Normalization modes: {ctx['normalization_modes']}")
    tpl.append("")

    # QC block
    tpl.append("QC / Dataset summary")
    tpl.append("--------------------")
    tpl.append(f"Samples: {ctx.get('n_samples', 'na')}, Features: {ctx.get('n_features', 'na')}")
    for k, v in ctx.items():
        if k.endswith("_counts") and hasattr(v, "to_string"):
            tpl.append(f"{k.replace('_counts', '')} counts:\n{v.to_string()}")
    if "const_features" in ctx:
        tpl.append(f"Constant features: {', '.join(ctx['const_features'][:5])}")
    if "validation_metrics" in ctx:
        vm = ctx["validation_metrics"]
        tpl.append(f"Validation balanced accuracy: {vm.get('balanced_accuracy', 'na')}")
        tpl.append(f"Per-class recall: {vm.get('per_class_recall', [])}")
    if warnings:
        tpl.append("Warnings:")
        for w in warnings:
            tpl.append(f"- {w}")
    tpl.append("")

    tpl.append("RQ1 – Oil discrimination & clustering")
    tpl.append("-------------------------------------")
    tpl.append("")

    tpl.append("RQ2 – Stability (CV %)")
    tpl.append("----------------------")
    top_stable = (
        stability[stability["level"] == "overall"].sort_values("cv_percent").head(top_k)[["feature", "cv_percent"]]
    )
    tpl.append(top_stable.to_string(index=False))
    tpl.append("")

    tpl.append("RQ3 – Discriminative power (ANOVA / Kruskal)")
    tpl.append("--------------------------------------------")
    if not discrim.empty:
        tpl.append(
            discrim.sort_values("p_value")
            .head(top_k)[
                ["feature", "method", "statistic", "p_value"]
                + (["p_value_adj"] if "p_value_adj" in discrim.columns else [])
            ]
            .to_string(index=False)
        )
        if "p_value_adj" in discrim.columns:
            tpl.append("P-values adjusted by FDR (Benjamini–Hochberg).")
    else:
        tpl.append("No discriminative tests computed.")
    tpl.append("")

    if feat_importance is not None and not feat_importance.empty:
        tpl.append("Feature importance (classifier-based)")
        tpl.append("------------------------------------")
        tpl.append(feat_importance.head(top_k)[["feature", "rf_importance", "lr_coef_abs"]].to_string(index=False))
        tpl.append("")
    if norm_comp is not None and not norm_comp.empty:
        tpl.append("Normalization comparison")
        tpl.append("------------------------")
        tpl.append(norm_comp.to_string(index=False))
        tpl.append("")

    tpl.append("RQ4 – Heating trends")
    tpl.append("---------------------")
    if not heating.empty:
        tpl.append(
            heating.sort_values("p_value")
            .head(top_k)[
                ["feature", "slope", "p_value", "monotonic_trend"]
                + (["p_value_adj"] if "p_value_adj" in heating.columns else [])
            ]
            .to_string(index=False)
        )
        if "p_value_adj" in heating.columns:
            tpl.append("Trend p-values adjusted by FDR.")
    else:
        tpl.append("No heating trends computed.")
    tpl.append("")

    tpl.append("Oil vs chips divergence")
    tpl.append("-----------------------")
    if not oil_vs_chips.empty:
        diverging = oil_vs_chips[oil_vs_chips["diverges"]]
        if diverging.empty:
            tpl.append("No strong divergences detected.")
        else:
            tpl.append("Top divergent markers (mean/Trend/CV differences):")
            cols = [
                "feature",
                "p_mean",
                "p_mean_adj" if "p_mean_adj" in oil_vs_chips.columns else None,
                "cohen_d",
                "delta_cv",
                "delta_slope",
                "interpretation",
            ]
            cols = [c for c in cols if c is not None and c in oil_vs_chips.columns]
            tpl.append(diverging.head(top_k)[cols].to_string(index=False))
    else:
        tpl.append("Matrix column not provided; no comparison run.")

    if minimal_panel is not None and not minimal_panel.empty:
        tpl.append("")
        tpl.append("Minimal marker panel")
        tpl.append("---------------------")
        tpl.append(minimal_panel.to_string(index=False))
        if "status" in minimal_panel.columns and (minimal_panel["status"] == "not_met").any():
            tpl.append("Target accuracy not met; best available panel shown.")

    if clustering_metrics:
        tpl.append("")
        tpl.append("Clustering structure")
        tpl.append("--------------------")
        for k, v in clustering_metrics.items():
            tpl.append(f"{k}: {v}")

    if warnings:
        tpl.append("")
        tpl.append("Warnings / Guardrails")
        tpl.append("---------------------")
        tpl.extend(warnings)

    return "\n".join(tpl) + "\n"
