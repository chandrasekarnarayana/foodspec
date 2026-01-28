"""Protocol step for model training + validation (classification or regression)."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from foodspec.modeling.api import fit_predict
from foodspec.modeling.outcome import OutcomeType
from foodspec.qc.regression_diagnostics import summarize_regression_diagnostics

from .base import Step


class ModelFitPredictStep(Step):
    """Run model fit + predict using the modeling API."""

    name = "model_fit_predict"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def _select_features(self, df: pd.DataFrame, *, target: str, drop_cols: List[str]) -> List[str]:
        if self.cfg.get("feature_columns"):
            return [c for c in self.cfg.get("feature_columns", []) if c in df.columns]
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        return [c for c in numeric_cols if c not in drop_cols and c != target]

    def run(self, ctx: Dict[str, Any]):
        df: pd.DataFrame = ctx.get("data")
        if df is None:
            raise ValueError("model_fit_predict step requires a DataFrame in ctx['data'].")

        proto_cfg = ctx.get("config")
        outcome = OutcomeType(self.cfg.get("outcome_type", getattr(proto_cfg, "outcome_type", "classification")))
        target_col = (
            self.cfg.get("target_column")
            or getattr(proto_cfg, "target_column", None)
            or (getattr(proto_cfg, "expected_columns", {}) or {}).get("target_column")
        )
        if not target_col:
            raise ValueError("target_column is required for model_fit_predict step.")
        if target_col not in df.columns:
            raise ValueError(f"target_column '{target_col}' not found in dataset.")

        event_col = getattr(proto_cfg, "event_column", None)
        time_col = getattr(proto_cfg, "time_column", None)
        drop_cols = [c for c in [target_col, event_col, time_col] if c]
        drop_cols.extend(self.cfg.get("exclude_columns", []))
        feature_cols = self._select_features(df, target=target_col, drop_cols=drop_cols)
        if not feature_cols:
            raise ValueError("No feature columns available for modeling step.")

        group_col = self.cfg.get("group_column") or (getattr(proto_cfg, "expected_columns", {}) or {}).get("batch_col")
        groups = df[group_col].to_numpy() if group_col and group_col in df.columns else None

        X = df[feature_cols].astype(float).to_numpy()
        y = df[target_col].to_numpy()

        scheme = self.cfg.get("scheme") or getattr(proto_cfg, "validation_strategy", "nested")
        if scheme == "standard":
            scheme = "nested"

        model_name = self.cfg.get("model")
        if not model_name:
            model_name = (
                "ridge"
                if outcome == OutcomeType.REGRESSION
                else "poisson"
                if outcome == OutcomeType.COUNT
                else "lightgbm"
            )

        result = fit_predict(
            X,
            y,
            model_name=model_name,
            scheme=scheme,
            groups=groups,
            outer_splits=self.cfg.get("outer_splits", 5),
            inner_splits=self.cfg.get("inner_splits", 3),
            seed=self.cfg.get("seed", getattr(proto_cfg, "seed", 0)),
            allow_random_cv=self.cfg.get("allow_random_cv", False),
            param_grid=self.cfg.get("param_grid"),
            outcome_type=outcome,
            embedding=self.cfg.get("embedding"),
        )

        # Store artifacts
        ctx["tables"]["model_metrics"] = pd.DataFrame([result.metrics])
        if result.metrics_ci:
            ctx["tables"]["model_metrics_ci"] = (
                pd.DataFrame(result.metrics_ci).T.reset_index().rename(columns={"index": "metric"})
            )
        ctx["tables"]["model_predictions"] = pd.DataFrame({"y_true": result.y_true, "y_pred": result.y_pred})

        if outcome in {OutcomeType.REGRESSION, OutcomeType.COUNT}:
            diag = summarize_regression_diagnostics(result.y_true, result.y_pred, outcome_type=outcome)
            ctx["tables"]["model_residuals"] = diag["residuals"]
            ctx.setdefault("metadata", {}).setdefault("diagnostics", {})["model"] = diag["summary"]
            if diag["flags"]:
                ctx["logs"].append(f"[model_fit_predict] qc flags: {', '.join(diag['flags'])}")

        ctx.setdefault("metadata", {}).setdefault("modeling", {})["fit_predict"] = {
            "model": model_name,
            "outcome_type": outcome.value,
            "scheme": scheme,
            "features": feature_cols,
            "target": target_col,
        }
        embed_diag = result.diagnostics.get("embedding") if hasattr(result, "diagnostics") else None
        if embed_diag:
            ctx.setdefault("metadata", {}).setdefault("modeling", {})["embedding"] = embed_diag
        ctx["logs"].append(f"[model_fit_predict] completed ({outcome.value})")
