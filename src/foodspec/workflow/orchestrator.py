"""End-to-end orchestration for FoodSpec CLI.

Coordinates validation, preprocessing, feature extraction, modeling,
trust evaluation, visualization, and reporting into a single call.
The protocol file provides defaults; CLI flags can override modeling,
scheme, features, and reporting options without mutating the protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.manifest import RunManifest
from foodspec.core.errors import FoodSpecValidationError
from foodspec.features.hybrid import extract_features
from foodspec.features.schema import FeatureConfig, FeatureInfo, parse_feature_config, split_spectral_dataframe
from foodspec.io.validators import validate_input
from foodspec.modeling.api import fit_predict
from foodspec.reporting.api import build_report_from_run
from foodspec.trust.evaluator import TrustEvaluator
from foodspec.utils.run_artifacts import init_run_dir, write_run_summary, get_logger
from foodspec.protocol.config import ProtocolConfig
from foodspec.protocol.utils import validate_protocol


@dataclass
class OrchestratorResult:
    status: str
    artifacts: Dict[str, Any]
    details: Dict[str, Any]


class EndToEndOrchestrator:
    """High-level orchestrator used by `foodspec run-e2e`."""

    def __init__(
        self,
        csv_path: Path,
        protocol_path: Path,
        out_dir: Path,
        *,
        scheme: str,
        model: str,
        feature_type: str,
        label_col: Optional[str],
        group_col: Optional[str],
        mode: str,
        enable_trust: bool = True,
        enable_viz: bool = True,
        enable_report: bool = True,
        generate_pdf: bool = False,
        seed: int = 0,
        unsafe_random_cv: bool = False,
    ):
        self.csv_path = csv_path
        self.protocol_path = protocol_path
        self.out_dir = out_dir
        self.scheme = scheme
        self.model = model
        self.feature_type = feature_type
        self.label_col = label_col
        self.group_col = group_col
        self.mode = mode
        self.enable_trust = enable_trust
        self.enable_viz = enable_viz
        self.enable_report = enable_report
        self.generate_pdf = generate_pdf
        self.seed = seed
        self.unsafe_random_cv = unsafe_random_cv

    def _resolve_columns(self, df: pd.DataFrame, cfg: ProtocolConfig) -> Tuple[str, Optional[str]]:
        label = self.label_col
        if label is None and cfg.expected_columns:
            label = cfg.expected_columns.get("label_col") or cfg.expected_columns.get("oil_col")
        if label is None and "label" in df.columns:
            label = "label"
        if label is None:
            raise FoodSpecValidationError("Label column not provided; use --label-col or protocol expected_columns.label_col.")
        if label not in df.columns:
            raise FoodSpecValidationError(f"Label column '{label}' not found in CSV.")

        group = self.group_col
        if group is None:
            if "stage" in df.columns:
                group = "stage"
            elif "batch" in df.columns:
                group = "batch"
        if group is not None and group not in df.columns:
            raise FoodSpecValidationError(f"Group column '{group}' not found in CSV.")
        return label, group

    def _choose_scheme(self, groups: Optional[np.ndarray]) -> str:
        # Respect explicit scheme, else infer based on presence of groups
        scheme = (self.scheme or "").lower()
        if scheme:
            return scheme
        if groups is not None:
            return "loso"
        return "random"

    def _plot_confusion(self, y_true: np.ndarray, y_pred: np.ndarray, classes: Iterable[str], registry: ArtifactRegistry) -> Optional[Path]:
        try:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right")
            ax.set_yticklabels(classes)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            path = registry.save_interpretability_plot(fig, "confusion_matrix.png")
            plt.close(fig)
            return path
        except Exception:
            return None

    def _run_trust_stack(
        self,
        registry: ArtifactRegistry,
        model,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        # Simple stratified split for calibration/test; fallback to all data if tiny
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(y))
        if len(np.unique(y)) > 1 and len(y) >= 6:
            skf = StratifiedKFold(n_splits=min(3, len(np.unique(y))), shuffle=True, random_state=self.seed)
            train_idx, test_idx = next(iter(skf.split(X, y)))
        else:
            rng.shuffle(idx)
            split = max(1, int(0.2 * len(y)))
            test_idx, train_idx = idx[:split], idx[split:]
        X_cal, y_cal = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        evaluator = TrustEvaluator(model, registry, target_coverage=0.9, abstention_threshold=0.7, random_state=self.seed)
        try:
            evaluator.fit_conformal(X_cal, y_cal)
            if hasattr(model, "predict_proba"):
                proba_cal = model.predict_proba(X_cal)
                evaluator.fit_calibration(y_cal, proba_cal)
            result = evaluator.evaluate(
                X_test,
                y_test,
                batch_ids=groups[test_idx] if groups is not None else None,
                model_name=str(model.__class__.__name__),
            )
        except Exception as exc:
            raise FoodSpecValidationError(f"Trust evaluation failed: {exc}")

        registry.write_json(registry.trust_eval_path, asdict(result))
        registry.write_json(registry.calibration_path, {"method": evaluator.calibration_method, "target_coverage": evaluator.target_coverage})
        registry.write_trust_calibration_metrics({"ece": result.ece, "coverage": result.conformal_coverage})
        return asdict(result)

    def _extract_features(
        self,
        df: pd.DataFrame,
        cfg: ProtocolConfig,
        feature_config: FeatureConfig,
        label_col: str,
        group_col: Optional[str],
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], list[FeatureInfo], dict]:
        X, wavenumbers, meta = split_spectral_dataframe(df, exclude=[label_col, group_col])
        labels = meta[label_col].to_numpy()
        groups = meta[group_col].to_numpy() if group_col and group_col in meta.columns else None
        feats_df, info, details = extract_features(
            X,
            wavenumbers,
            feature_type=self.feature_type,
            config=feature_config,
            labels=labels,
            seed=self.seed,
        )
        return feats_df, labels, groups, info, details

    def run(self) -> OrchestratorResult:
        run_dir = init_run_dir(self.out_dir)
        logger = get_logger(run_dir)
        registry = ArtifactRegistry(run_dir)
        registry.ensure_layout()

        start_time = datetime.now(timezone.utc)

        try:
            if not self.csv_path.exists():
                raise FoodSpecValidationError(f"CSV not found: {self.csv_path}")
            if not self.protocol_path.exists():
                raise FoodSpecValidationError(f"Protocol not found: {self.protocol_path}")

            df = pd.read_csv(self.csv_path)
            proto_cfg = ProtocolConfig.from_file(self.protocol_path)
            diag = validate_input(self.csv_path)
            proto_diag = validate_protocol(proto_cfg, df)
            errors = (diag.get("errors") or []) + (proto_diag.get("errors") or [])
            if errors:
                raise FoodSpecValidationError("; ".join(errors))

            label_col, group_col = self._resolve_columns(df, proto_cfg)
            scheme = self._choose_scheme(df[group_col].to_numpy() if group_col else None)

            feature_config = parse_feature_config(self.protocol_path)
            feats_df, labels, groups, info, details = self._extract_features(df, proto_cfg, feature_config, label_col, group_col)

            model_result = fit_predict(
                feats_df.to_numpy(dtype=float),
                labels,
                model_name=self.model,
                scheme=scheme,
                groups=groups,
                seed=self.seed,
                allow_random_cv=self.unsafe_random_cv,
            )

            metrics_rows = [
                {"metric": k, "value": v} for k, v in model_result.metrics.items()
            ]
            registry.write_csv(registry.metrics_path, metrics_rows)
            folds_rows = []
            for fold in model_result.folds:
                row = {"fold": fold.get("fold"), **fold.get("metrics", {})}
                folds_rows.append(row)
            if folds_rows:
                registry.write_csv(registry.metrics_per_fold_path, folds_rows)

            preds_rows = []
            for yt, yp in zip(model_result.y_true, model_result.y_pred):
                preds_rows.append({"y_true": int(yt), "y_pred": int(yp)})
            registry.write_csv(registry.predictions_path, preds_rows)

            trust_payload = {}
            if self.enable_trust and hasattr(model_result.model, "predict_proba"):
                labels_for_trust = labels
                if model_result.classes:
                    mapping = {cls: idx for idx, cls in enumerate(model_result.classes)}
                    try:
                        labels_for_trust = np.array([mapping[label] for label in labels])
                    except KeyError:
                        labels_for_trust = labels
                trust_payload = self._run_trust_stack(
                    registry,
                    model_result.model,
                    feats_df.to_numpy(dtype=float),
                    labels_for_trust,
                    groups,
                )

            viz_path = None
            if self.enable_viz:
                viz_path = self._plot_confusion(model_result.y_true, model_result.y_pred, model_result.classes, registry)

            report_artifacts: Dict[str, str] = {}
            artifacts = {
                "metrics": str(registry.metrics_path),
                "metrics_per_fold": str(registry.metrics_per_fold_path),
                "predictions": str(registry.predictions_path),
                "features": str(run_dir / "features.csv"),
                "feature_info": str(run_dir / "feature_info.json"),
                "trust": str(registry.trust_eval_path) if trust_payload else None,
                "report": None,
                "report_pdf": None,
                "viz_confusion": str(viz_path) if viz_path else None,
            }

            # Build manifest compatible with ReportContext / RunManifest (pre-report)
            protocol_snapshot = asdict(proto_cfg)
            pre_report_manifest = RunManifest.build(
                protocol_snapshot=protocol_snapshot,
                data_path=self.csv_path,
                seed=self.seed,
                artifacts={k: v for k, v in artifacts.items() if v},
                validation_spec={"scheme": scheme, "model": self.model, "features": self.feature_type},
                trust_config={"enabled": self.enable_trust, "target_coverage": 0.9, "abstention_threshold": 0.7},
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
            )
            pre_report_manifest.save(registry.manifest_path)

            # Reporting
            if self.enable_report:
                report_artifacts = build_report_from_run(
                    run_dir,
                    out_dir=run_dir,
                    mode=self.mode,
                    pdf=self.generate_pdf,
                    title=f"FoodSpec Report ({self.mode})",
                )
                artifacts["report"] = report_artifacts.get("report_html")
                artifacts["report_pdf"] = report_artifacts.get("report_pdf")

            # Save final manifest with report artifacts included
            final_manifest = RunManifest.build(
                protocol_snapshot=protocol_snapshot,
                data_path=self.csv_path,
                seed=self.seed,
                artifacts={k: v for k, v in artifacts.items() if v},
                validation_spec={"scheme": scheme, "model": self.model, "features": self.feature_type},
                trust_config={"enabled": self.enable_trust, "target_coverage": 0.9, "abstention_threshold": 0.7},
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
            )
            final_manifest.save(registry.manifest_path)

            # Persist feature table and metadata for traceability
            feats_df.to_csv(run_dir / "features.csv", index=False)
            feature_info_serialized = [entry.to_dict() if isinstance(entry, FeatureInfo) else dict(entry) for entry in info]
            (run_dir / "feature_info.json").write_text(json.dumps({"features": feature_info_serialized}, indent=2))

            run_summary = {
                "status": "success",
                "scheme": scheme,
                "model": self.model,
                "features": self.feature_type,
                "label_col": label_col,
                "group_col": group_col,
                "metrics": model_result.metrics,
                "metrics_ci": model_result.metrics_ci,
                "trust": trust_payload,
                "artifacts": artifacts,
                "seed": self.seed,
            }
            write_run_summary(run_dir, run_summary)
            return OrchestratorResult(status="success", artifacts=artifacts, details=run_summary)

        except FoodSpecValidationError as exc:
            write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
            logger.error(str(exc))
            return OrchestratorResult(status="validation_error", artifacts={}, details={"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive guard
            write_run_summary(run_dir, {"status": "fail", "error": str(exc)})
            logger.error("Runtime error in run-e2e: %s", exc)
            return OrchestratorResult(status="runtime_error", artifacts={}, details={"error": str(exc)})


__all__ = ["EndToEndOrchestrator", "OrchestratorResult"]
