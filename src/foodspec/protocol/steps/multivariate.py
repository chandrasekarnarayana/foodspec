"""Protocol step for multivariate analysis (PCA, LDA, MDS, stats)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from foodspec.multivariate import MultivariateComponent, build_component
from foodspec.qc.multivariate import (
    MultivariateQCPolicy,
    batch_drift,
    compute_pca_outlier_scores,
    hotelling_t2,
    outlier_flags,
    summarize_scores,
)

from .base import Step


class MultivariateAnalysisStep(Step):
    """Run a multivariate component and emit protocol artifacts."""

    name = "multivariate_analysis"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _select_features(self, df: pd.DataFrame, drop_cols: List[str]) -> List[str]:
        if self.cfg.get("feature_columns"):
            return [c for c in self.cfg.get("feature_columns", []) if c in df.columns and c not in drop_cols]
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        return [c for c in numeric_cols if c not in drop_cols]

    def _resolve_label_column(self, df: pd.DataFrame, proto_cfg: Any) -> Optional[str]:
        label_col = self.cfg.get("label_column") or getattr(proto_cfg, "target_column", None)
        if not label_col:
            label_col = (getattr(proto_cfg, "expected_columns", {}) or {}).get("target_column")
        if label_col and label_col not in df.columns:
            raise ValueError(f"label_column '{label_col}' not found in dataset")
        return label_col

    def run(self, ctx: Dict[str, Any]):
        df: pd.DataFrame = ctx.get("data")
        if df is None:
            raise ValueError("multivariate_analysis step requires a DataFrame in ctx['data'].")

        proto_cfg = ctx.get("config")
        method = self.cfg.get("method", "pca")
        params = self.cfg.get("params", {})
        label_col = self._resolve_label_column(df, proto_cfg)
        group_col = self.cfg.get("group_column") or (getattr(proto_cfg, "expected_columns", {}) or {}).get("batch_col")
        view2_cols = self.cfg.get("view2_columns", [])

        drop_cols: List[str] = []
        if label_col:
            drop_cols.append(label_col)
        drop_cols.extend(self.cfg.get("exclude_columns", []))
        drop_cols.extend(view2_cols)
        feature_cols = self._select_features(df, drop_cols)
        if not feature_cols:
            raise ValueError("No usable feature columns for multivariate analysis.")

        X = df[feature_cols].astype(float).to_numpy()
        sample_ids = list(df.index)
        component: MultivariateComponent = build_component(method, **params)

        y_array: Optional[np.ndarray] = None
        paired_view: Optional[np.ndarray] = None
        if getattr(component, "requires_second_view", False):
            if not view2_cols:
                raise ValueError("view2_columns are required for paired-view methods (e.g., CCA).")
            paired_view = df[view2_cols].astype(float).to_numpy()
        if getattr(component, "requires_y", False):
            if not label_col:
                raise ValueError(f"Method '{method}' requires label_column or target_column.")
            y_array = df[label_col].to_numpy()

        if paired_view is not None:
            result = component.fit_transform(X, paired_view)
        else:
            result = component.fit_transform(X, y_array)

        comp_names = [f"{method}_c{i+1}" for i in range(result.scores.shape[1])]
        scores_df = pd.DataFrame(result.scores, columns=comp_names)
        scores_df.insert(0, "sample_id", sample_ids)
        ctx.setdefault("tables", {})["multivariate_scores"] = scores_df

        if result.loadings is not None:
            loadings_df = pd.DataFrame(result.loadings, columns=feature_cols)
            loadings_df.insert(0, "component", comp_names)
            ctx.setdefault("tables", {})["multivariate_loadings"] = loadings_df

        summary = summarize_scores(result.scores)
        if result.explained_variance is not None:
            summary["explained_variance_sum"] = float(np.sum(result.explained_variance))
        summary.update({"method": method, "n_features": len(feature_cols)})
        ctx.setdefault("tables", {})["multivariate_summary"] = pd.DataFrame([summary])

        qc_cfg = (self.cfg.get("qc") or {}).copy()
        proto_qc = getattr(proto_cfg, "qc", {}) or {}
        if not qc_cfg and isinstance(proto_qc, dict):
            qc_cfg = proto_qc.get("multivariate", {}) or {}

        qc_enabled = bool(qc_cfg.get("enabled", False))
        qc_tables = ctx.setdefault("tables", {})
        qc_artifacts = ctx.setdefault("qc_artifacts", {})
        qc_summary: Dict[str, Any] = {}

        if qc_enabled:
            outlier_cfg = qc_cfg.get("outliers", {}) if isinstance(qc_cfg, dict) else {}
            policy = MultivariateQCPolicy.from_dict(outlier_cfg)
            method_name = outlier_cfg.get("method", "hotelling_t2")
            thresholds: Dict[str, float] = {}
            score_dict: Dict[str, np.ndarray] = {}

            if method_name == "hotelling_t2":
                alpha = float(outlier_cfg.get("alpha", policy.alpha))
                cov_mode = "robust" if outlier_cfg.get("covariance", "empirical") == "robust" else "empirical"
                t2, limit = hotelling_t2(result.scores, covariance=cov_mode, alpha=alpha)
                thresholds["t2"] = limit
                score_dict["t2"] = t2
            else:
                pca_method = outlier_cfg.get("method", "robust")
                pca_res = compute_pca_outlier_scores(result.scores, method="robust" if pca_method == "robust" else "classic")
                score_dict["score_distance"] = np.asarray(pca_res["score_distance"], dtype=float)
                score_dict["orthogonal_distance"] = np.asarray(pca_res["orthogonal_distance"], dtype=float)
                thresholds.update(pca_res.get("thresholds", {}))

            # Override thresholds based on policy strategy when requested
            strategy = policy.threshold_strategy
            if strategy == "quantile":
                for key, vals in score_dict.items():
                    thresholds[key] = float(np.quantile(vals, policy.quantile))
            elif strategy == "mad":
                for key, vals in score_dict.items():
                    med = float(np.median(vals))
                    mad_val = float(stats.median_abs_deviation(vals, scale=1.0))
                    thresholds[key] = med + policy.mad_multiplier * mad_val

            outlier_table = outlier_flags(sample_ids, score_dict, thresholds, policy)
            qc_tables["multivariate_qc"] = outlier_table
            qc_artifacts["multivariate_outliers"] = outlier_table

            drift_cfg = qc_cfg.get("drift", {}) if isinstance(qc_cfg, dict) else {}
            drift_enabled = bool(drift_cfg.get("enabled", False))
            drift_df = None
            if drift_enabled and group_col and group_col in df.columns:
                drift_df = batch_drift(
                    result.scores,
                    df[group_col].to_numpy(),
                    metric=drift_cfg.get("metric", "centroid_l2"),
                    warn_threshold=float(drift_cfg.get("warn_threshold", 2.0)),
                    fail_threshold=float(drift_cfg.get("fail_threshold", 4.0)),
                )
                if not drift_df.empty:
                    qc_tables["multivariate_group_shift"] = drift_df
                    qc_artifacts["multivariate_drift"] = drift_df

            flagged = outlier_table[outlier_table["flag"]]
            status = "pass"
            if not flagged.empty:
                status = "warn" if policy.severity == "warn" else "fail"
            if drift_df is not None and not drift_df.empty:
                if (drift_df["status"] == "fail").any():
                    status = "fail"
                elif status == "pass" and (drift_df["status"] == "warn").any():
                    status = "warn"
            mv_summary = {
                "status": status,
                "outliers": {
                    "n_flagged": int(flagged.shape[0]),
                    "thresholds": thresholds,
                    "policy": policy.__dict__,
                },
            }
            if drift_df is not None and not drift_df.empty:
                mv_summary["drift"] = drift_df.to_dict(orient="records")
            qc_summary = {"multivariate": mv_summary}
            qc_artifacts["qc_summary"] = qc_summary

        meta_slot = ctx.setdefault("metadata", {}).setdefault("multivariate", {})
        meta_slot.update({"method": method, "params": params, "qc": qc_summary, "features": feature_cols})
        if label_col:
            meta_slot["label_column"] = label_col
        if group_col and group_col in df.columns:
            meta_slot["group_column"] = group_col

        try:
            import matplotlib.pyplot as plt

            if result.scores.shape[1] >= 2:
                fig, ax = plt.subplots()
                color = df[label_col].to_numpy() if label_col else None
                scatter = ax.scatter(result.scores[:, 0], result.scores[:, 1], c=color, cmap="viridis", alpha=0.8)
                if qc_enabled and "multivariate_outliers" in qc_artifacts:
                    qc_flags = qc_artifacts["multivariate_outliers"]
                    flagged_mask = qc_flags["flag"].to_numpy(dtype=bool)
                    ax.scatter(
                        result.scores[flagged_mask, 0],
                        result.scores[flagged_mask, 1],
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.5,
                        label="QC outlier",
                    )
                ax.set_xlabel(comp_names[0])
                ax.set_ylabel(comp_names[1])
                ax.set_title(f"{method.upper()} scores")
                if label_col:
                    legend1 = ax.legend(*scatter.legend_elements(), title=label_col)
                    ax.add_artist(legend1)
                if qc_enabled and "multivariate_outliers" in qc_artifacts:
                    ax.legend(loc="best")
                ctx.setdefault("figures", {})["multivariate/scores"] = fig

                if qc_enabled and "multivariate_drift" in qc_artifacts:
                    drift_plot = plt.figure()
                    drift_ax = drift_plot.add_subplot(1, 1, 1)
                    drift_df_plot = qc_artifacts["multivariate_drift"]
                    drift_ax.bar(drift_df_plot["group"].astype(str), drift_df_plot["value"], color="#2563eb")
                    drift_ax.axhline(float(qc_cfg.get("drift", {}).get("warn_threshold", 2.0)), color="#f59e0b", linestyle="--", label="warn")
                    drift_ax.axhline(float(qc_cfg.get("drift", {}).get("fail_threshold", 4.0)), color="#ef4444", linestyle=":", label="fail")
                    drift_ax.set_ylabel("Centroid shift (L2)")
                    drift_ax.set_title("Batch drift")
                    drift_ax.legend()
                    qc_artifacts.setdefault("qc_figures", {})["multivariate_drift"] = drift_plot
        except Exception:
            ctx.setdefault("logs", []).append("[multivariate] plotting skipped (matplotlib unavailable)")

        ctx.setdefault("logs", []).append(f"[multivariate] {method} completed on {len(feature_cols)} features")