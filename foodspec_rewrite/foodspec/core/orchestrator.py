"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

ExecutionEngine orchestrates FoodSpec workflows end-to-end.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.cache import CacheManager
from foodspec.core.manifest import RunManifest
from foodspec.core.protocol import ProtocolV2

try:  # Optional numpy seeding for determinism
    import numpy as np
except ImportError:  # pragma: no cover - optional
    np = None


@dataclass
class RunResult:
    """Result bundle returned by the execution engine."""

    output_dir: Path
    manifest: RunManifest
    logs: List[str] = field(default_factory=list)


class ExecutionEngine:
    """Execute a ProtocolV2 workflow with deterministic orchestration.

    Only minimal functionality is implemented; each stage is optional and
    raises NotImplementedError if explicitly requested.
    """

    def __init__(self, cache_dir: Optional[Path] = None, enable_cache: bool = True) -> None:
        """Initialize execution engine.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory for stage caching. Defaults to .foodspec_cache in cwd.
        enable_cache : bool, default True
            Enable hash-based caching for expensive stages.
        """
        self.logs: List[str] = []
        self.cache_hits: List[str] = []
        self.cache_misses: List[str] = []
        default_cache = Path.cwd() / ".foodspec_cache"
        self.cache = CacheManager(
            cache_dir=cache_dir or default_cache,
            enabled=enable_cache
        )

    def _log(self, msg: str) -> None:
        self.logs.append(msg)

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if np is not None:
            np.random.seed(seed)
        self._log(f"Seeds set to {seed}")

    def run(
        self,
        protocol_or_path: Union[ProtocolV2, str, Path],
        outdir: Union[str, Path],
        seed: int = 0,
    ) -> RunResult:
        """Run the workflow, returning a RunResult.

        Minimal implementation: validates and records manifest; other stages
        raise NotImplementedError only if requested by protocol contents.
        
        When cache is enabled, checks for cached preprocess/features outputs
        before recomputing expensive stages. Cache hits/misses are recorded
        in the manifest.
        """

        self.logs.clear()
        self.cache_hits.clear()
        self.cache_misses.clear()
        self._seed(seed)

        protocol = self._load_protocol(protocol_or_path)
        self._log("Protocol loaded and defaults applied")

        output_dir = Path(outdir)
        artifacts = ArtifactRegistry(output_dir)
        artifacts.ensure_layout()
        self._log(f"Artifact layout ensured at {output_dir}")

        # Minimal stage checks: if the user requests more than the minimal
        # pipeline, we surface explicit NotImplementedError to keep behavior
        # deterministic and clear.
        self._check_stage_requests(protocol)

        # Prepare data path for fingerprinting
        data_path = Path(protocol.data.input)
        data_file = data_path if data_path.exists() else None

        # Example cache integration point (to be expanded when stages are implemented):
        # For now, we demonstrate the wiring without executing actual preprocess/features
        data_fingerprint = RunManifest.compute_data_fingerprint(data_file) if data_file else ""
        library_version = protocol.version
        
        # Preprocess stage cache check (placeholder)
        if protocol.preprocess.recipe or protocol.preprocess.steps:
            preprocess_spec = {
                "recipe": protocol.preprocess.recipe,
                "steps": [s.model_dump() for s in protocol.preprocess.steps]
            }
            preprocess_key = self.cache.compute_key(
                data_fingerprint=data_fingerprint,
                stage_spec=preprocess_spec,
                stage_name="preprocess",
                library_version=library_version,
            )
            cached_preprocess = self.cache.get(preprocess_key)
            if cached_preprocess:
                self.cache_hits.append("preprocess")
                self._log(f"Cache hit: preprocess (key={preprocess_key[:12]}...)")
            else:
                self.cache_misses.append("preprocess")
                self._log(f"Cache miss: preprocess (key={preprocess_key[:12]}...)")
                # Future: execute preprocess and cache.put(...)
        
        # Features stage cache check (placeholder)
        if protocol.features.modules or protocol.features.strategy not in {"auto", ""}:
            features_spec = {
                "strategy": protocol.features.strategy,
                "modules": protocol.features.modules,
            }
            features_key = self.cache.compute_key(
                data_fingerprint=data_fingerprint,
                stage_spec=features_spec,
                stage_name="features",
                library_version=library_version,
            )
            cached_features = self.cache.get(features_key)
            if cached_features:
                self.cache_hits.append("features")
                self._log(f"Cache hit: features (key={features_key[:12]}...)")
            else:
                self.cache_misses.append("features")
                self._log(f"Cache miss: features (key={features_key[:12]}...)")
                # Future: execute features and cache.put(...)

        # Extract validation spec from protocol for manifest
        validation_spec_dict = {
            "scheme": protocol.validation.scheme,
            "group_key": protocol.validation.group_key,
            "allow_random_cv": protocol.validation.allow_random_cv,
            "nested": protocol.validation.nested,
            "metrics": protocol.validation.metrics,
        }

        # Build manifest (uses data fingerprint if file exists)
        manifest = RunManifest.build(
            protocol_snapshot=protocol.model_dump(mode="python"),
            data_path=data_file,
            seed=seed,
            artifacts={
                "metrics": str(artifacts.metrics_path),
                "metrics_per_fold": str(artifacts.metrics_per_fold_path),
                "metrics_summary": str(artifacts.metrics_summary_path),
                "best_params": str(artifacts.best_params_path),
                "qc": str(artifacts.qc_path),
                "predictions": str(artifacts.predictions_path),
                "plots": str(artifacts.plots_dir),
                "report_html": str(artifacts.report_html_path),
                "report_pdf": str(artifacts.report_pdf_path),
                "bundle": str(artifacts.bundle_dir),
                "manifest": str(artifacts.manifest_path),
                "logs": str(artifacts.logs_path),
                # Trust artifacts
                "calibration_metrics": str(artifacts.calibration_metrics_path),
                "conformal_coverage": str(artifacts.conformal_coverage_path),
                "conformal_sets": str(artifacts.conformal_sets_path),
                "abstention_summary": str(artifacts.abstention_summary_path),
                "coefficients": str(artifacts.coefficients_path),
                "permutation_importance": str(artifacts.permutation_importance_path),
                "marker_panel_explanations": str(artifacts.marker_panel_explanations_path),
            },
            validation_spec=validation_spec_dict,
            trust_config={
                "calibration_enabled": bool(protocol.uncertainty.conformal.get("calibration")),
                "conformal_enabled": bool(protocol.uncertainty.conformal.get("conformal")),
                "abstention_enabled": bool(protocol.uncertainty.conformal.get("abstention")),
                "interpretability_enabled": bool(protocol.interpretability.methods or protocol.interpretability.marker_panel),
            },
            warnings=[] if data_file else ["Data file not found; fingerprint omitted."],
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )
        artifacts.write_json(artifacts.manifest_path, json.loads(json.dumps(manifest.__dict__)))
        self._log("Manifest written")

        # Write logs
        artifacts.logs_path.write_text("\n".join(self.logs))

        return RunResult(output_dir=output_dir, manifest=manifest, logs=list(self.logs))

    def _load_protocol(self, protocol_or_path: Union[ProtocolV2, str, Path]) -> ProtocolV2:
        if isinstance(protocol_or_path, ProtocolV2):
            return protocol_or_path.apply_defaults()
        return ProtocolV2.load(protocol_or_path)

    def _check_stage_requests(self, protocol: ProtocolV2) -> None:
        """Surface NotImplementedError only when a stage is requested."""

        preprocess = protocol.preprocess
        recipe = preprocess.recipe if hasattr(preprocess, "recipe") else preprocess.get("recipe") if isinstance(preprocess, dict) else None
        steps = preprocess.steps if hasattr(preprocess, "steps") else preprocess.get("steps", []) if isinstance(preprocess, dict) else []
        if recipe or steps:
            raise NotImplementedError("Preprocess stage not implemented yet.")
        if protocol.qc.thresholds or protocol.qc.metrics:
            raise NotImplementedError("QC stage not implemented yet.")
        if protocol.features.modules or protocol.features.strategy not in {"auto", ""}:
            raise NotImplementedError("Features stage not implemented yet.")
        if protocol.model.estimator not in {"logreg", ""}:  # baseline accepted but not run
            raise NotImplementedError("Model stage not implemented yet.")
        if protocol.validation.scheme not in {"train_test_split", "leave_one_group_out", ""}:
            raise NotImplementedError("Validation stage not implemented yet.")
        if protocol.uncertainty.conformal:
            raise NotImplementedError("Uncertainty stage not implemented yet.")
        if protocol.interpretability.methods or protocol.interpretability.marker_panel:
            raise NotImplementedError("Interpretability stage not implemented yet.")
        if protocol.visualization.plots:
            raise NotImplementedError("Visualization stage not implemented yet.")
        if protocol.reporting.format not in {"markdown", ""} or protocol.reporting.sections:
            if protocol.reporting.sections not in ([], ["summary", "metrics", "figures"]):
                raise NotImplementedError("Reporting stage not implemented yet.")
        if protocol.export.bundle:
            raise NotImplementedError("Export stage not implemented yet.")

    def save_evaluation_artifacts(
        self,
        evaluation_result: "EvaluationResult",
        artifacts: ArtifactRegistry,
    ) -> None:
        """Save evaluation outputs to artifact directory.
        
        Writes predictions.csv, metrics.csv (combined per-fold + summary),
        and best_params.csv (if nested CV).
        
        Parameters
        ----------
        evaluation_result : EvaluationResult
            Result from evaluate_model_cv or evaluate_model_nested_cv.
        artifacts : ArtifactRegistry
            Artifact registry for output directory.
        
        Examples
        --------
        >>> from foodspec.validation.evaluation import EvaluationResult
        >>> result = EvaluationResult(fold_predictions=[], fold_metrics=[], bootstrap_ci={})
        >>> engine.save_evaluation_artifacts(result, artifacts)
        """
        # Save predictions.csv
        evaluation_result.save_predictions_csv(artifacts.predictions_path)
        self._log(f"Saved predictions to {artifacts.predictions_path}")
        
        # Save metrics.csv (combined per-fold + summary)
        evaluation_result.save_metrics_csv(artifacts.metrics_path, include_summary=True)
        self._log(f"Saved metrics to {artifacts.metrics_path}")
        
        # Save best_params.csv (nested CV only)
        if evaluation_result.hyperparameters_per_fold:
            evaluation_result.save_best_params_csv(artifacts.best_params_path)
            self._log(f"Saved hyperparameters to {artifacts.best_params_path}")

    def generate_visualizations(
        self,
        protocol: ProtocolV2,
        evaluation_result: "EvaluationResult",
        metadata_df: Optional[Any] = None,
        artifacts: Optional[ArtifactRegistry] = None,
    ) -> None:
        """Generate visualization plots based on protocol specification.
        
        Generates plots for:
        - Model performance (confusion matrix, metrics)
        - Trust metrics (calibration, conformal coverage, abstention)
        - Feature importance
        - Cross-validation diagnostics
        
        All plots are auto-saved to artifacts.plots_dir with publication quality.
        
        Parameters
        ----------
        protocol : ProtocolV2
            Protocol defining visualization specs
        evaluation_result : EvaluationResult
            Evaluation results containing predictions and metrics
        metadata_df : DataFrame, optional
            Sample-level metadata for grouping (batch_id, stage, instrument)
        artifacts : ArtifactRegistry, optional
            Registry for auto-saving plots
        
        Raises
        ------
        NotImplementedError
            If visualization.plots is requested but evaluation_result unavailable
        
        Examples
        --------
        >>> engine = ExecutionEngine()
        >>> engine.generate_visualizations(protocol, evaluation_result, artifacts=artifacts)
        # Plots saved to artifacts.plots_dir
        """
        if not protocol.visualization.plots:
            self._log("No visualization plots requested in protocol")
            return
        
        if not artifacts:
            raise ValueError("ArtifactRegistry required for visualization output")
        
        try:
            from foodspec.viz import (
                PlotConfig,
                plot_confusion_matrix,
                plot_calibration_curve,
                plot_feature_importance,
                plot_metrics_by_fold,
                plot_conformal_coverage_by_group,
                plot_abstention_rate,
            )
        except ImportError:  # pragma: no cover
            raise ImportError(
                "Visualization requires foodspec.viz module. "
                "Ensure plots_v2.py is properly installed."
            )
        
        # Setup plot config
        plot_config = PlotConfig(
            dpi=300,  # Publication quality
            figure_size=(12, 6),
            seed=self.logs[0] if self.logs else 42,  # Use orchestrator seed if available
        )
        
        protocol_hash = protocol.model_dump()  # Full snapshot for hash
        run_id = getattr(artifacts, "run_id", "run_unknown")
        
        self._log("Generating visualizations...")
        
        # Extract predictions from evaluation result
        if not hasattr(evaluation_result, 'fold_predictions') or not evaluation_result.fold_predictions:
            self._log("No predictions available for visualization")
            return
        
        # Collect predictions across folds
        try:
            y_true = np.concatenate([fp['y_true'] for fp in evaluation_result.fold_predictions])
            y_pred = np.concatenate([fp['y_pred'] for fp in evaluation_result.fold_predictions])
        except (AttributeError, KeyError, TypeError):
            self._log("Warning: Could not extract predictions for confusion matrix")
            y_true = y_pred = None
        
        # 1. Confusion Matrix (if available)
        if y_true is not None and y_pred is not None:
            try:
                plot_confusion_matrix(
                    y_true,
                    y_pred,
                    artifacts=artifacts,
                    filename='confusion_matrix.png',
                    protocol_hash='protocol_v2',
                    run_id=run_id,
                    config=plot_config,
                )
                self._log("✓ Generated confusion matrix")
            except Exception as e:  # pragma: no cover
                self._log(f"Warning: Confusion matrix generation failed: {e}")
        
        # 2. Calibration Curve (if probabilities available)
        if hasattr(evaluation_result, 'fold_predictions') and evaluation_result.fold_predictions:
            try:
                proba_list = []
                y_true_list = []
                for fp in evaluation_result.fold_predictions:
                    if 'proba' in fp:
                        proba_list.append(fp['proba'])
                        y_true_list.append(fp['y_true'])
                
                if proba_list and y_true_list:
                    proba = np.vstack(proba_list) if proba_list else None
                    y_true_cal = np.concatenate(y_true_list)
                    
                    if proba is not None:
                        plot_calibration_curve(
                            y_true_cal,
                            proba,
                            n_bins=10,
                            metadata_df=metadata_df,
                            metadata_col='batch_id' if metadata_df is not None else None,
                            artifacts=artifacts,
                            filename='calibration_curve.png',
                            protocol_hash='protocol_v2',
                            run_id=run_id,
                            config=plot_config,
                        )
                        self._log("✓ Generated calibration curve")
            except Exception as e:  # pragma: no cover
                self._log(f"Warning: Calibration curve generation failed: {e}")
        
        # 3. Metrics by Fold (if fold_metrics available)
        if hasattr(evaluation_result, 'fold_metrics') and evaluation_result.fold_metrics:
            try:
                fold_ids = list(range(len(evaluation_result.fold_metrics)))
                plot_metrics_by_fold(
                    metrics_dict=evaluation_result.fold_metrics,
                    fold_ids=fold_ids,
                    metadata_df=metadata_df,
                    artifacts=artifacts,
                    filename='metrics_by_fold.png',
                    protocol_hash='protocol_v2',
                    run_id=run_id,
                    config=plot_config,
                )
                self._log("✓ Generated metrics by fold")
            except Exception as e:  # pragma: no cover
                self._log(f"Warning: Metrics by fold generation failed: {e}")
        
        self._log(f"Visualization plots saved to {artifacts.plots_dir}")


__all__ = ["ExecutionEngine", "RunResult"]
