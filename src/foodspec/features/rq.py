"""
Ratio-Quality (RQ) Engine
=========================

Lightweight, dependency-minimal analysis of Raman/FTIR peak ratios for oils/chips.
The engine is DataFrame-first (no file I/O) and returns structured results that
can be rendered in GUI/report layers.

Example
-------
>>> import pandas as pd
>>> from foodspec.features.rq import (
...     PeakDefinition, RatioDefinition, RQConfig, RatioQualityEngine
... )
>>> df = pd.DataFrame({
...     "sample_id": [1, 2, 3, 4],
...     "oil_type": ["A", "A", "B", "B"],
...     "matrix": ["oil", "oil", "oil", "oil"],
...     "heating_stage": [0, 1, 0, 1],
...     "I_1742": [10, 9, 6, 5],
...     "I_1652": [4, 3, 8, 7],
... })
>>> peaks = [
...     PeakDefinition(name="I_1742", column="I_1742"),
...     PeakDefinition(name="I_1652", column="I_1652"),
... ]
>>> ratios = [RatioDefinition(name="1742/1652", numerator="I_1742", denominator="I_1652")]
>>> cfg = RQConfig(oil_col="oil_type", matrix_col="matrix", heating_col="heating_stage")
>>> engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=cfg)
>>> results = engine.run_all(df)
>>> print(results.text_report[:120])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------


@dataclass
class PeakDefinition:
    """Named peak/intensity column."""

    name: str
    column: str
    wavenumber: Optional[float] = None
    window: Optional[Tuple[float, float]] = None  # Optional metadata for UI/reporting
    mode: str = "max"  # max | area


@dataclass
class RatioDefinition:
    """Numerator / denominator ratio built from PeakDefinition names or raw columns."""

    name: str
    numerator: str
    denominator: str


@dataclass
class RQConfig:
    """Column naming conventions and options."""

    oil_col: str = "Oil_Name"
    matrix_col: str = "matrix"
    heating_col: str = "Heating_Stage"
    replicate_col: str = "replicate_id"
    sample_col: str = "sample_id"
    # Whether to include classifier-based feature importance
    compute_feature_importance: bool = True
    random_state: int = 0
    n_splits: int = 5
    normalization_modes: List[str] = field(default_factory=lambda: ["reference"])
    minimal_panel_target_accuracy: float = 0.9
    enable_clustering: bool = True
    adjust_p_values: bool = True
    max_features: Optional[int] = None


@dataclass
class RatioQualityResult:
    ratio_table: pd.DataFrame
    stability_summary: pd.DataFrame
    discriminative_summary: pd.DataFrame
    feature_importance: Optional[pd.DataFrame]
    heating_trend_summary: pd.DataFrame
    oil_vs_chips_summary: pd.DataFrame
    normalization_comparison: Optional[pd.DataFrame]
    minimal_panel: Optional[pd.DataFrame]
    clustering_metrics: Optional[Dict]
    warnings: List[str]
    text_report: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _cv(series: pd.Series) -> Dict[str, float]:
    vals = series.dropna().astype(float)
    if len(vals) == 0:
        return {"mean": np.nan, "std": np.nan, "cv_percent": np.nan, "mad": np.nan, "mad_over_mean": np.nan}
    mean = vals.mean()
    std = vals.std(ddof=1) if len(vals) > 1 else 0.0
    mad = (vals - mean).abs().median()
    return {
        "mean": mean,
        "std": std,
        "cv_percent": (std / mean * 100) if mean != 0 else np.nan,
        "mad": mad,
        "mad_over_mean": (mad / mean) if mean != 0 else np.nan,
    }


def _safe_group_vectors(df: pd.DataFrame, group_col: str, feature: str) -> List[np.ndarray]:
    vectors = []
    for _, sub in df.groupby(group_col):
        vec = sub[feature].dropna().to_numpy(dtype=float)
        if len(vec) > 1:
            vectors.append(vec)
    return vectors


def _monotonic_label(slope: float, p_value: float, alpha: float = 0.05) -> str:
    if np.isnan(p_value):
        return "no clear trend"
    if p_value >= alpha:
        return "no clear trend"
    return "increases with heating" if slope > 0 else "decreases with heating"


def _rf_accuracy(df: pd.DataFrame, features: List[str], label_col: str, random_state: int, n_splits: int) -> float:
    X = df[features].astype(float)
    y = df[label_col].astype(str)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]
    class_counts = y.value_counts(dropna=True)
    if len(class_counts) < 2 or X.empty:
        return np.nan
    min_class = class_counts.min()
    if min_class < 2:
        return np.nan
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    cv_splits = max(2, min(int(min_class), len(y), len(class_counts), n_splits))
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    scores = cross_val_score(rf, Xz, y, cv=cv)
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class RatioQualityEngine:
    """
    Compute ratio quality metrics (stability, discrimination, heating trends, oil-vs-chips).

    The engine expects a tidy DataFrame with metadata columns (oil/matrix/heating)
    and intensity columns referenced by PeakDefinition/RatioDefinition.
    """

    def __init__(
        self,
        peaks: Iterable[PeakDefinition],
        ratios: Iterable[RatioDefinition],
        config: Optional[RQConfig] = None,
    ):
        self.peaks = list(peaks)
        self.ratios = list(ratios)
        self.config = config or RQConfig()

        self._peak_map = {p.name: p for p in self.peaks}
        self._ratio_map = {r.name: r for r in self.ratios}

    # ----------------------------------------------
    # Public API
    # ----------------------------------------------
    def run_all(self, df: pd.DataFrame, validation_metrics: Optional[Dict[str, Any]] = None) -> RatioQualityResult:
        warnings: List[str] = []
        qc_context: Dict[str, Any] = {}
        # QC: counts and constants
        n_samples = len(df)
        feature_cols = self._feature_columns(candidate_df=df)
        qc_context["n_samples"] = n_samples
        qc_context["n_features"] = len(feature_cols)
        cv_allowed = True
        for label_col in [self.config.oil_col, self.config.matrix_col, self.config.heating_col]:
            if label_col in df.columns:
                counts = df[label_col].value_counts(dropna=True)
                qc_context[f"{label_col}_counts"] = counts
                min_count = counts.min()
                if pd.notna(min_count) and min_count < 2:
                    cv_allowed = False
                    warnings.append(
                        f"Label '{label_col}' has <2 samples in a class; cross-validation metrics will be skipped."
                    )
                if counts.min() < max(2, self.config.n_splits):
                    warnings.append(f"Label '{label_col}' has very few samples in at least one class ({counts.min()}).")
                if counts.nunique() < 2:
                    warnings.append(f"Label '{label_col}' has <2 classes; related analyses may be skipped.")
        const_feats = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
        if const_feats:
            warnings.append(f"Constant/degenerate features dropped from consideration: {', '.join(const_feats[:5])}")
            qc_context["const_features"] = const_feats

        df_ratios = self.compute_ratios(df)
        stability = self.compute_stability(df_ratios)
        discrim, feat_imp = (None, None)
        if cv_allowed:
            discrim, feat_imp = self.compute_discriminative_power(df_ratios, cv_allowed=cv_allowed)
        heating = self.compute_heating_trends(df_ratios)
        oil_vs_chips = self.compare_oil_vs_chips(df_ratios)
        norm_comp = self.compare_normalizations(df)
        minimal_panel = self.compute_minimal_panel(df_ratios, feat_imp) if cv_allowed else None
        clustering = self.compute_clustering_metrics(df_ratios) if cv_allowed else None
        # normalize Nones to empty DataFrames for downstream text rendering
        if not cv_allowed:
            warnings.append("Cross-validation skipped due to too few samples per class.")
        if discrim is None:
            discrim = pd.DataFrame()
        if feat_imp is None:
            feat_imp = pd.DataFrame()
        if norm_comp is None:
            norm_comp = pd.DataFrame()
        if heating is None:
            heating = pd.DataFrame()
        if oil_vs_chips is None:
            oil_vs_chips = pd.DataFrame()
        if minimal_panel is None:
            minimal_panel = pd.DataFrame()
        if clustering is None:
            clustering = {}
        # Guardrails
        n_samples, n_features = df_ratios.shape
        if self.config.max_features and n_features > self.config.max_features:
            warnings.append(
                f"Feature count {n_features} exceeds max_features {self.config.max_features}; results may be slow."
            )
        if n_samples < 2 * self.config.n_splits:
            warnings.append(f"Low sample count ({n_samples}) relative to CV splits ({self.config.n_splits}).")
        if n_features > 5 * max(1, n_samples):
            warnings.append(
                f"High feature-to-sample ratio (features {n_features} vs samples {n_samples}); interpret cautiously."
            )
        if minimal_panel is not None and "status" in minimal_panel.columns:
            if (minimal_panel["status"] == "not_met").any():
                warnings.append("Minimal panel target accuracy not met; best available panel reported.")
        qc_context["warnings"] = warnings.copy()
        if validation_metrics:
            qc_context["validation_metrics"] = validation_metrics

        report = self.generate_text_report(
            stability,
            discrim,
            heating,
            oil_vs_chips,
            feat_imp,
            norm_comp,
            minimal_panel,
            clustering,
            warnings,
            context=qc_context,
        )
        return RatioQualityResult(
            ratio_table=df_ratios,
            stability_summary=stability,
            discriminative_summary=discrim,
            feature_importance=feat_imp,
            heating_trend_summary=heating,
            oil_vs_chips_summary=oil_vs_chips,
            normalization_comparison=norm_comp,
            minimal_panel=minimal_panel,
            clustering_metrics=clustering,
            warnings=warnings,
            text_report=report,
        )

    def compute_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for r in self.ratios:
            if r.numerator not in df_out.columns or r.denominator not in df_out.columns:
                df_out[r.name] = np.nan
                continue
            num = df_out[r.numerator].astype(float)
            den = df_out[r.denominator].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df_out[r.name] = num / den
        return df_out

    def compute_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._feature_columns()
        rows = []
        # Global CV
        for feat in features:
            metrics = _cv(df[feat])
            rows.append({"feature": feat, "level": "overall", "group": "", **metrics})
        # Per-oil CV
        if self.config.oil_col in df.columns:
            for oil, sub in df.groupby(self.config.oil_col):
                for feat in features:
                    metrics = _cv(sub[feat])
                    rows.append({"feature": feat, "level": "oil", "group": str(oil), **metrics})
        return pd.DataFrame(rows)

    def compute_discriminative_power(
        self, df: pd.DataFrame, cv_allowed: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        features = self._feature_columns()
        oil_col = self.config.oil_col
        discrim_rows = []

        if oil_col in df.columns:
            for feat in features:
                groups = _safe_group_vectors(df, oil_col, feat)
                if len(groups) < 2:
                    continue
                try:
                    f_stat, p_val = stats.f_oneway(*groups)
                    method = "ANOVA"
                    stat_val = f_stat
                except Exception:
                    h_stat, p_val = stats.kruskal(*groups)
                    method = "Kruskal"
                    stat_val = h_stat
                discrim_rows.append(
                    {
                        "feature": feat,
                        "method": method,
                        "statistic": float(stat_val),
                        "p_value": float(p_val),
                    }
                )

        if self.config.adjust_p_values and discrim_rows:
            pvals = [r["p_value"] for r in discrim_rows]
            reject, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
            for r, adj, rej in zip(discrim_rows, p_adj, reject):
                r["p_value_adj"] = float(adj)
                r["significant_fdr"] = bool(rej)

        feat_importance = None
        if self.config.compute_feature_importance and oil_col in df.columns and cv_allowed:
            feat_importance = self._compute_feature_importance(df, features, oil_col)

        discrim_df = pd.DataFrame(discrim_rows)
        return discrim_df, feat_importance

    def compute_heating_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._feature_columns()
        heating_col = self.config.heating_col
        if heating_col not in df.columns:
            return pd.DataFrame(columns=["feature", "slope", "p_value", "r_squared", "monotonic_trend"])

        rows = []
        for feat in features:
            sub = df[[feat, heating_col]].dropna()
            if sub.empty:
                continue
            x = pd.to_numeric(sub[heating_col], errors="coerce")
            y = sub[feat].astype(float)
            mask = ~x.isna() & ~y.isna()
            x = x[mask]
            y = y[mask]
            if len(x) < 3:
                continue
            slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
            rho, p_spear = stats.spearmanr(x, y)
            rows.append(
                {
                    "feature": feat,
                    "slope": slope,
                    "p_value": p_val,
                    "r_squared": r_val**2,
                    "p_spearman": p_spear,
                    "spearman_rho": rho,
                    "monotonic_trend": _monotonic_label(slope, p_val),
                }
            )
        if self.config.adjust_p_values and rows:
            pvals = [r["p_value"] for r in rows]
            reject, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
            for r, adj, rej in zip(rows, p_adj, reject):
                r["p_value_adj"] = float(adj)
                r["significant_fdr"] = bool(rej)
        return pd.DataFrame(rows)

    def compare_oil_vs_chips(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare stability and heating trends between matrix types (oil vs chips).
        Flags divergence when slopes differ in significance/sign.
        """
        matrix_col = self.config.matrix_col
        heating_col = self.config.heating_col
        features = self._feature_columns()

        if matrix_col not in df.columns:
            return pd.DataFrame(columns=["feature", "oil_cv", "chips_cv", "oil_slope", "chips_slope", "diverges"])

        rows = []
        for feat in features:
            oil_sub = df[df[matrix_col] == "oil"]
            chips_sub = df[df[matrix_col] == "chips"]

            oil_cv = _cv(oil_sub[feat])["cv_percent"] if not oil_sub.empty else np.nan
            chips_cv = _cv(chips_sub[feat])["cv_percent"] if not chips_sub.empty else np.nan

            oil_slope, oil_p = self._trend(oil_sub, heating_col, feat)
            chips_slope, chips_p = self._trend(chips_sub, heating_col, feat)

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
            def _spearman(sub):
                if heating_col not in sub.columns:
                    return (np.nan, np.nan)
                x = pd.to_numeric(sub[heating_col], errors="coerce")
                y = sub[feat].astype(float)
                mask = ~x.isna() & ~y.isna()
                if mask.sum() < 3:
                    return (np.nan, np.nan)
                rho, pval = stats.spearmanr(x[mask], y[mask])
                return (rho, pval)

            oil_rho, oil_rho_p = _spearman(oil_sub)
            chips_rho, chips_rho_p = _spearman(chips_sub)

            # Divergence criteria
            diverges = False
            # trend divergence: significance mismatch or opposite sign
            if np.isfinite(oil_p) and np.isfinite(chips_p):
                sig_oil = oil_p < 0.05
                sig_chips = chips_p < 0.05
                if sig_oil != sig_chips:
                    diverges = True
                elif sig_oil and sig_chips and np.sign(oil_slope) != np.sign(chips_slope):
                    diverges = True
            # mean divergence
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
        if self.config.adjust_p_values and not df_out.empty:
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
        tags = []
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
        df_out["interpretation"] = tags
        if not df_out.empty:
            df_out["diverges"] = pd.Series([bool(x) for x in df_out["diverges"]], dtype=object)
        return df_out

    def generate_text_report(
        self,
        stability: pd.DataFrame,
        discrim: pd.DataFrame,
        heating: pd.DataFrame,
        oil_vs_chips: pd.DataFrame,
        feat_importance: Optional[pd.DataFrame],
        norm_comp: Optional[pd.DataFrame],
        minimal_panel: Optional[pd.DataFrame],
        clustering_metrics: Optional[Dict],
        warnings: List[str],
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> str:
        """
        Generate a text report with optional protocol/context fields.
        context keys: protocol_name, intro, preprocessing_notes, normalization_modes.
        """
        ctx = context or {}
        tpl = []
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
                tpl.append(f"{k.replace('_counts','')} counts:\n{v.to_string()}")
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

    # ----------------------------------------------
    # Internal helpers
    # ----------------------------------------------
    def _feature_columns(self, candidate_df: Optional[pd.DataFrame] = None) -> List[str]:
        peak_cols = [p.column for p in self.peaks if p.column]
        ratio_cols = [r.name for r in self.ratios]
        cols = peak_cols + ratio_cols
        if candidate_df is not None:
            cols = [c for c in cols if c in candidate_df.columns]
        return cols

    def _compute_feature_importance(self, df: pd.DataFrame, features: List[str], label_col: str) -> pd.DataFrame:
        X = df[features].astype(float)
        y = df[label_col].astype(str)
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        class_counts = y.value_counts(dropna=True)
        if X.empty or len(class_counts) < 2 or class_counts.min() < 2:
            return pd.DataFrame(columns=["feature", "rf_importance", "lr_coef_abs", "lr_cv_accuracy", "rf_cv_accuracy"])

        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)

        cv = StratifiedKFold(
            n_splits=max(2, min(self.config.n_splits, int(class_counts.min()), len(y))),
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Logistic regression (multiclass)
        lr = LogisticRegression(max_iter=2000)
        lr_scores = cross_val_score(lr, Xz, y, cv=cv)
        lr.fit(Xz, y)
        lr_coef = np.abs(lr.coef_).mean(axis=0)

        # Random forest
        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        rf_scores = cross_val_score(rf, Xz, y, cv=cv)
        rf.fit(Xz, y)
        rf_importances = rf.feature_importances_

        out = pd.DataFrame(
            {
                "feature": features,
                "rf_importance": rf_importances,
                "lr_coef_abs": lr_coef,
            }
        ).sort_values("rf_importance", ascending=False)
        out["lr_cv_accuracy"] = lr_scores.mean()
        out["rf_cv_accuracy"] = rf_scores.mean()
        return out

    def _trend(self, df: pd.DataFrame, heating_col: str, feat: str) -> Tuple[float, float]:
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
        return (slope, p_val)

    # ----------------------------------------------
    # Extensions: normalization comparison, minimal panel, clustering
    # ----------------------------------------------
    def compare_normalizations(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        modes = self.config.normalization_modes or ["reference"]
        oil_col = self.config.oil_col
        raw_peaks = [p.column for p in self.peaks if p.column in df.columns]
        if not raw_peaks or oil_col not in df.columns:
            return None
        rows = []
        for mode in modes:
            tmp = df.copy()
            if mode == "reference":
                ref_col = raw_peaks[0]
                norms = tmp[ref_col].replace(0, np.nan).astype(float)
                feat_cols = []
                for col in raw_peaks:
                    tmp[f"{col}_{mode}"] = tmp[col].astype(float) / norms
                    if col != ref_col:
                        feat_cols.append(f"{col}_{mode}")
            else:
                X = tmp[raw_peaks].astype(float).to_numpy()
                if mode == "vector":
                    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                elif mode == "area":
                    Xn = X / (np.sum(X, axis=1, keepdims=True) + 1e-12)
                elif mode == "max":
                    Xn = X / (np.max(X, axis=1, keepdims=True) + 1e-12)
                else:
                    continue
                feat_cols = [f"{c}_{mode}" for c in raw_peaks]
                tmp_feats = pd.DataFrame(Xn, columns=feat_cols, index=tmp.index)
                tmp = pd.concat([tmp, tmp_feats], axis=1)
            acc = _rf_accuracy(tmp, feat_cols, oil_col, self.config.random_state, self.config.n_splits)
            rows.append({"norm_method": mode, "rf_cv_accuracy": acc, "n_features": len(feat_cols)})
        return pd.DataFrame(rows)

    def compute_minimal_panel(
        self,
        df: pd.DataFrame,
        feat_importance: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        oil_col = self.config.oil_col
        ratio_cols = [c for c in df.columns if c.startswith("I_") or "/" in c or "_norm" in c]
        if oil_col not in df.columns or not ratio_cols:
            return None
        X = df[ratio_cols].astype(float)
        y = df[oil_col].astype(str)
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        if len(y.unique()) < 2 or X.empty:
            return None
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)
        # Target accuracy guardrail
        target_acc = self.config.minimal_panel_target_accuracy

        lr = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=2000,
            C=1.0,
            random_state=self.config.random_state,
        )
        lr.fit(Xz, y)
        coef_mag = np.abs(lr.coef_).mean(axis=0)
        selected_idx = np.where(coef_mag > 1e-5)[0]
        if len(selected_idx) == 0:
            return None
        selected_feats = [ratio_cols[i] for i in selected_idx]
        acc = _rf_accuracy(df.loc[mask, :], selected_feats, oil_col, self.config.random_state, self.config.n_splits)
        status = "met" if (not np.isnan(acc) and acc >= target_acc) else "not_met"
        panel_df = pd.DataFrame(
            {
                "selected_features": [", ".join(selected_feats)],
                "n_features": [len(selected_feats)],
                "rf_cv_accuracy": [acc],
                "target_accuracy": [target_acc],
                "status": [status],
            }
        )
        return panel_df

    def compute_clustering_metrics(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self.config.enable_clustering:
            return None
        oil_col = self.config.oil_col
        feat_cols = [c for c in df.columns if c not in [oil_col, self.config.matrix_col, self.config.heating_col]]
        if oil_col not in df.columns or not feat_cols:
            return None
        X = df[feat_cols].astype(float)
        y = df[oil_col].astype(str)
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        if len(y.unique()) < 2:
            return None
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)
        n_clusters = len(np.unique(y))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init=10)
        labels = kmeans.fit_predict(Xz)
        sil = silhouette_score(Xz, labels)
        ari = adjusted_rand_score(y, labels)
        return {"silhouette": float(sil), "ari_vs_oil": float(ari), "n_clusters": int(n_clusters)}


__all__ = [
    "PeakDefinition",
    "RatioDefinition",
    "RQConfig",
    "RatioQualityEngine",
    "RatioQualityResult",
]
