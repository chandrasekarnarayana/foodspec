"""
Ratio-Quality analysis step for protocol execution.

Runs the RatioQualityEngine to perform stability, discriminative power,
and heating trend analysis on spectral peak ratios.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from foodspec.features.rq import PeakDefinition, RatioDefinition, RatioQualityEngine, RQConfig

from .base import Step


class RQAnalysisStep(Step):
    """Perform Ratio-Quality analysis on spectral data."""

    name = "rq_analysis"

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize RQ analysis step.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Step configuration with peaks, ratios, and analysis parameters.
        """
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        """Execute RQ analysis on data in context.

        Parameters
        ----------
        ctx : Dict[str, Any]
            Execution context with 'data' DataFrame.
        """
        peaks_cfg = self.cfg.get("peaks", [])
        ratios_cfg = self.cfg.get("ratios", [])
        peaks = [
            PeakDefinition(
                name=p["name"],
                column=p.get("column", p["name"]),
                wavenumber=p.get("wavenumber"),
                window=tuple(p.get("window")) if p.get("window") else None,
            )
            for p in peaks_cfg
        ]
        ratios = [
            RatioDefinition(
                name=r["name"],
                numerator=r["numerator"],
                denominator=r["denominator"],
            )
            for r in ratios_cfg
        ]
        rq_cfg = RQConfig(
            oil_col=self.cfg.get("oil_col", "oil_type"),
            matrix_col=self.cfg.get("matrix_col", "matrix"),
            heating_col=self.cfg.get("heating_col", "heating_stage"),
            random_state=self.cfg.get("random_state", 0),
            n_splits=self.cfg.get("n_splits", 5),
            normalization_modes=self.cfg.get("normalization_modes", ["reference"]),
            minimal_panel_target_accuracy=self.cfg.get("minimal_panel_target_accuracy", 0.9),
            enable_clustering=self.cfg.get("enable_clustering", True),
            adjust_p_values=self.cfg.get("adjust_p_values", True),
        )
        engine = RatioQualityEngine(peaks=peaks, ratios=ratios, config=rq_cfg)
        # Optional validation metrics
        validation_metrics = None
        try:
            from foodspec.validation import nested_cv

            df = ctx["data"]
            feature_cols = [p.column for p in peaks] + [r.name for r in ratios]
            feature_cols = [c for c in feature_cols if c in df.columns]
            label_col = rq_cfg.oil_col
            if feature_cols and label_col in df.columns:
                X = df[feature_cols].astype(float).to_numpy()
                y = df[label_col].astype(str).to_numpy()
                class_counts = pd.Series(y).value_counts()
                if (class_counts < 2).any():
                    validation_metrics = None
                    raise ValueError("Too few samples per class for CV; skipping validation metrics.")
                groups = None
                if self.cfg.get("validation_strategy") == "batch_aware" and self.cfg.get("batch_col") in df.columns:
                    groups = df[self.cfg.get("batch_col")].to_numpy()
                results = nested_cv(
                    RandomForestClassifier(n_estimators=150, random_state=rq_cfg.random_state),
                    X,
                    y,
                    groups=groups,
                    outer_splits=max(2, min(5, len(np.unique(y)))),
                    inner_splits=3,
                )
                if results:
                    # Aggregate metrics
                    bal_acc = float(np.mean([r["bal_accuracy"] for r in results]))
                    recalls = np.mean([r["per_class_recall"] for r in results], axis=0).tolist()
                    aucs = [r.get("roc_auc") for r in results if r.get("roc_auc") is not None]
                    validation_metrics = {
                        "balanced_accuracy": bal_acc,
                        "per_class_recall": recalls,
                        "roc_auc": float(np.mean(aucs)) if aucs else None,
                    }
        except Exception:
            validation_metrics = None

        res = engine.run_all(ctx["data"], validation_metrics=validation_metrics)
        ctx["tables"].update(
            {
                "stability_summary": res.stability_summary,
                "discriminative_summary": res.discriminative_summary,
                "feature_importance": res.feature_importance,
                "heating_trend_summary": res.heating_trend_summary,
                "oil_vs_chips_summary": res.oil_vs_chips_summary,
                "normalization_comparison": res.normalization_comparison,
                "minimal_panel": res.minimal_panel,
            }
        )
        ctx["figures"].update(res.figures if hasattr(res, "figures") else {})
        ctx["report"] = res.text_report
        ctx["summary"] = " ".join(res.text_report.splitlines()[:10])
        ctx["logs"].append("[rq_analysis] completed")
