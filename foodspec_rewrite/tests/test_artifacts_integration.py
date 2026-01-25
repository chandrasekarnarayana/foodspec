"""
Tests for Phase 7 - Artifacts Integration.

Verifies that evaluation outputs are properly saved to disk:
- predictions.csv with y_true, y_pred, proba columns, fold_id, group
- metrics.csv with per-fold metrics and summary statistics
- best_params.csv when nested CV is enabled
- RunManifest includes validation_spec details
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.manifest import RunManifest
from foodspec.core.orchestrator import ExecutionEngine
from foodspec.validation.evaluation import EvaluationResult


class TestEvaluationResultArtifacts:
    """Test EvaluationResult artifact saving methods."""

    def test_save_predictions_csv_basic(self, tmp_path):
        """Test predictions.csv saves correctly with all required columns."""
        fold_predictions = [
            {"fold_id": 0, "sample_idx": 0, "y_true": 0, "y_pred": 0, "proba_0": 0.8, "proba_1": 0.2},
            {"fold_id": 0, "sample_idx": 1, "y_true": 1, "y_pred": 1, "proba_0": 0.3, "proba_1": 0.7},
            {"fold_id": 1, "sample_idx": 2, "y_true": 0, "y_pred": 1, "proba_0": 0.4, "proba_1": 0.6},
        ]
        
        result = EvaluationResult(
            fold_predictions=fold_predictions,
            fold_metrics=[],
            bootstrap_ci={},
        )
        
        output_path = tmp_path / "predictions.csv"
        result.save_predictions_csv(output_path)
        
        assert output_path.exists()
        df = pd.read_csv(output_path)
        
        assert len(df) == 3
        assert list(df.columns) == ["fold_id", "sample_idx", "y_true", "y_pred", "proba_0", "proba_1"]
        assert df["fold_id"].tolist() == [0, 0, 1]
        assert df["y_true"].tolist() == [0, 1, 0]
        assert df["y_pred"].tolist() == [0, 1, 1]

    def test_save_predictions_csv_with_group(self, tmp_path):
        """Test predictions.csv includes group column for LOBO/LOGO."""
        fold_predictions = [
            {"fold_id": 0, "sample_idx": 0, "y_true": 0, "y_pred": 0, "proba_0": 0.8, "proba_1": 0.2, "group": "batch_A"},
            {"fold_id": 0, "sample_idx": 1, "y_true": 1, "y_pred": 1, "proba_0": 0.3, "proba_1": 0.7, "group": "batch_B"},
        ]
        
        result = EvaluationResult(
            fold_predictions=fold_predictions,
            fold_metrics=[],
            bootstrap_ci={},
        )
        
        output_path = tmp_path / "predictions.csv"
        result.save_predictions_csv(output_path)
        
        df = pd.read_csv(output_path)
        assert "group" in df.columns
        assert df["group"].tolist() == ["batch_A", "batch_B"]

    def test_save_predictions_csv_empty(self, tmp_path):
        """Test predictions.csv handles empty predictions."""
        result = EvaluationResult(
            fold_predictions=[],
            fold_metrics=[],
            bootstrap_ci={},
        )
        
        output_path = tmp_path / "predictions.csv"
        result.save_predictions_csv(output_path)
        
        assert output_path.exists()
        content = output_path.read_text()
        assert content == ""

    def test_save_metrics_csv_per_fold(self, tmp_path):
        """Test metrics.csv saves per-fold metrics."""
        fold_metrics = [
            {"fold_id": 0, "accuracy": 0.90, "macro_f1": 0.88, "auroc": 0.92},
            {"fold_id": 1, "accuracy": 0.85, "macro_f1": 0.83, "auroc": 0.87},
            {"fold_id": 2, "accuracy": 0.92, "macro_f1": 0.91, "auroc": 0.94},
        ]
        
        result = EvaluationResult(
            fold_predictions=[],
            fold_metrics=fold_metrics,
            bootstrap_ci={
                "accuracy": (0.85, 0.89, 0.92),
                "macro_f1": (0.83, 0.87, 0.91),
                "auroc": (0.87, 0.91, 0.94),
            },
        )
        
        output_path = tmp_path / "metrics.csv"
        result.save_metrics_csv(output_path, include_summary=False)
        
        df = pd.read_csv(output_path)
        assert len(df) == 3
        assert list(df.columns) == ["fold_id", "accuracy", "macro_f1", "auroc"]
        assert df["accuracy"].tolist() == [0.90, 0.85, 0.92]

    def test_save_metrics_csv_with_summary(self, tmp_path):
        """Test metrics.csv includes summary statistics (mean, std, CI)."""
        fold_metrics = [
            {"fold_id": 0, "accuracy": 0.90, "macro_f1": 0.88},
            {"fold_id": 1, "accuracy": 0.85, "macro_f1": 0.83},
            {"fold_id": 2, "accuracy": 0.92, "macro_f1": 0.91},
        ]
        
        result = EvaluationResult(
            fold_predictions=[],
            fold_metrics=fold_metrics,
            bootstrap_ci={
                "accuracy": (0.85, 0.89, 0.92),
                "macro_f1": (0.83, 0.87, 0.91),
            },
        )
        
        output_path = tmp_path / "metrics.csv"
        result.save_metrics_csv(output_path, include_summary=True)
        
        df = pd.read_csv(output_path)
        assert len(df) == 8  # 3 folds + mean + std + ci_lower + ci_median + ci_upper
        
        # Check summary rows exist
        summary_fold_ids = df["fold_id"].tolist()
        assert "mean" in summary_fold_ids
        assert "std" in summary_fold_ids
        assert "ci_lower" in summary_fold_ids
        assert "ci_median" in summary_fold_ids
        assert "ci_upper" in summary_fold_ids
        
        # Check mean accuracy
        mean_row = df[df["fold_id"] == "mean"]
        expected_mean = np.mean([0.90, 0.85, 0.92])
        assert abs(mean_row["accuracy"].values[0] - expected_mean) < 0.001
        
        # Check CI rows
        ci_lower_row = df[df["fold_id"] == "ci_lower"]
        assert abs(ci_lower_row["accuracy"].values[0] - 0.85) < 0.001

    def test_save_metrics_csv_empty(self, tmp_path):
        """Test metrics.csv handles empty metrics."""
        result = EvaluationResult(
            fold_predictions=[],
            fold_metrics=[],
            bootstrap_ci={},
        )
        
        output_path = tmp_path / "metrics.csv"
        result.save_metrics_csv(output_path)
        
        assert output_path.exists()
        content = output_path.read_text()
        assert content == ""

    def test_save_best_params_csv_nested_cv(self, tmp_path):
        """Test best_params.csv saves hyperparameters from nested CV."""
        hyperparameters_per_fold = [
            {"C": 1.0, "max_iter": 100},
            {"C": 10.0, "max_iter": 200},
            {"C": 0.1, "max_iter": 100},
        ]
        
        result = EvaluationResult(
            fold_predictions=[],
            fold_metrics=[],
            bootstrap_ci={},
            hyperparameters_per_fold=hyperparameters_per_fold,
        )
        
        output_path = tmp_path / "best_params.csv"
        result.save_best_params_csv(output_path)
        
        assert output_path.exists()
        df = pd.read_csv(output_path)
        
        assert len(df) == 3
        assert "fold_id" in df.columns
        assert "C" in df.columns
        assert "max_iter" in df.columns
        assert df["fold_id"].tolist() == [0, 1, 2]
        assert df["C"].tolist() == [1.0, 10.0, 0.1]

    def test_save_best_params_csv_empty(self, tmp_path):
        """Test best_params.csv handles no hyperparameters (standard CV)."""
        result = EvaluationResult(
            fold_predictions=[],
            fold_metrics=[],
            bootstrap_ci={},
            hyperparameters_per_fold=None,
        )
        
        output_path = tmp_path / "best_params.csv"
        result.save_best_params_csv(output_path)
        
        assert output_path.exists()
        content = output_path.read_text()
        assert content == ""


class TestArtifactRegistryPaths:
    """Test ArtifactRegistry includes new paths."""

    def test_artifact_registry_has_new_paths(self, tmp_path):
        """Test ArtifactRegistry exposes best_params, metrics_per_fold, metrics_summary paths."""
        registry = ArtifactRegistry(tmp_path)
        
        assert hasattr(registry, "best_params_path")
        assert hasattr(registry, "metrics_per_fold_path")
        assert hasattr(registry, "metrics_summary_path")
        
        assert registry.best_params_path == tmp_path / "best_params.csv"
        assert registry.metrics_per_fold_path == tmp_path / "metrics_per_fold.csv"
        assert registry.metrics_summary_path == tmp_path / "metrics_summary.csv"


class TestRunManifestValidationSpec:
    """Test RunManifest includes validation_spec field."""

    def test_manifest_includes_validation_spec(self, tmp_path):
        """Test RunManifest.build() accepts and stores validation_spec."""
        validation_spec = {
            "scheme": "leave_one_group_out",
            "n_splits": 5,
            "group_key": "batch",
            "allow_random_cv": False,
            "nested_cv": True,
        }
        
        manifest = RunManifest.build(
            protocol_snapshot={"version": "2.0.0"},
            data_path=None,
            seed=42,
            artifacts={"metrics": "metrics.csv"},
            validation_spec=validation_spec,
        )
        
        assert manifest.validation_spec == validation_spec
        assert manifest.validation_spec["scheme"] == "leave_one_group_out"
        assert manifest.validation_spec["n_splits"] == 5
        assert manifest.validation_spec["group_key"] == "batch"
        assert manifest.validation_spec["allow_random_cv"] is False
        assert manifest.validation_spec["nested_cv"] is True

    def test_manifest_validation_spec_empty_default(self, tmp_path):
        """Test RunManifest defaults to empty validation_spec if not provided."""
        manifest = RunManifest.build(
            protocol_snapshot={"version": "2.0.0"},
            data_path=None,
            seed=42,
            artifacts={"metrics": "metrics.csv"},
        )
        
        assert manifest.validation_spec == {}

    def test_manifest_saves_and_loads_validation_spec(self, tmp_path):
        """Test validation_spec persists through save/load cycle."""
        validation_spec = {
            "scheme": "k_fold",
            "n_splits": 10,
            "nested_cv": False,
        }
        
        manifest = RunManifest.build(
            protocol_snapshot={"version": "2.0.0"},
            data_path=None,
            seed=42,
            artifacts={"metrics": "metrics.csv"},
            validation_spec=validation_spec,
        )
        
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)
        
        loaded = RunManifest.load(manifest_path)
        assert loaded.validation_spec == validation_spec


class TestOrchestratorArtifactSaving:
    """Test ExecutionEngine saves evaluation artifacts."""

    def test_save_evaluation_artifacts_basic(self, tmp_path):
        """Test ExecutionEngine.save_evaluation_artifacts() saves all files."""
        engine = ExecutionEngine()
        artifacts = ArtifactRegistry(tmp_path)
        artifacts.ensure_layout()
        
        result = EvaluationResult(
            fold_predictions=[
                {"fold_id": 0, "sample_idx": 0, "y_true": 0, "y_pred": 0, "proba_0": 0.8, "proba_1": 0.2},
            ],
            fold_metrics=[
                {"fold_id": 0, "accuracy": 0.90, "macro_f1": 0.88},
            ],
            bootstrap_ci={"accuracy": (0.85, 0.89, 0.92)},
        )
        
        engine.save_evaluation_artifacts(result, artifacts)
        
        assert artifacts.predictions_path.exists()
        assert artifacts.metrics_path.exists()
        assert "Saved predictions" in engine.logs[-2]
        assert "Saved metrics" in engine.logs[-1]

    def test_save_evaluation_artifacts_with_nested_cv(self, tmp_path):
        """Test ExecutionEngine saves best_params.csv for nested CV."""
        engine = ExecutionEngine()
        artifacts = ArtifactRegistry(tmp_path)
        artifacts.ensure_layout()
        
        result = EvaluationResult(
            fold_predictions=[],
            fold_metrics=[],
            bootstrap_ci={},
            hyperparameters_per_fold=[{"C": 1.0}, {"C": 10.0}],
        )
        
        engine.save_evaluation_artifacts(result, artifacts)
        
        assert artifacts.best_params_path.exists()
        assert "Saved hyperparameters" in engine.logs[-1]
        
        df = pd.read_csv(artifacts.best_params_path)
        assert len(df) == 2

    def test_save_evaluation_artifacts_without_nested_cv(self, tmp_path):
        """Test ExecutionEngine skips best_params.csv for standard CV."""
        engine = ExecutionEngine()
        artifacts = ArtifactRegistry(tmp_path)
        artifacts.ensure_layout()
        
        result = EvaluationResult(
            fold_predictions=[],
            fold_metrics=[],
            bootstrap_ci={},
            hyperparameters_per_fold=None,
        )
        
        engine.save_evaluation_artifacts(result, artifacts)
        
        # best_params.csv should not be created
        assert not artifacts.best_params_path.exists()
        # Should not have hyperparameters log message
        assert not any("hyperparameters" in log for log in engine.logs)


class TestOrchestratorManifestIntegration:
    """Test ExecutionEngine includes validation_spec in manifest."""

    def test_orchestrator_includes_validation_spec_in_manifest(self, tmp_path):
        """Test ExecutionEngine.run() includes validation_spec in manifest."""
        from foodspec.core.protocol import ProtocolV2
        
        protocol = ProtocolV2(
            version="2.0.0",
            task={"name": "classification", "objective": "authentication"},
            data={"input": "nonexistent.csv", "modality": "raman", "label": "class"},
            validation={
                "scheme": "leave_one_group_out",
                "group_key": "batch",
                "allow_random_cv": False,
            },
        )
        
        engine = ExecutionEngine()
        result = engine.run(protocol, outdir=tmp_path, seed=42)
        
        manifest = result.manifest
        assert "validation_spec" in manifest.__dict__
        assert manifest.validation_spec["scheme"] == "leave_one_group_out"
        assert manifest.validation_spec["group_key"] == "batch"
        assert manifest.validation_spec["allow_random_cv"] is False

    def test_orchestrator_includes_artifact_paths_in_manifest(self, tmp_path):
        """Test ExecutionEngine.run() includes new artifact paths in manifest."""
        from foodspec.core.protocol import ProtocolV2
        
        protocol = ProtocolV2(
            version="2.0.0",
            task={"name": "classification", "objective": "authentication"},
            data={"input": "nonexistent.csv", "modality": "raman", "label": "class"},
        )
        
        engine = ExecutionEngine()
        result = engine.run(protocol, outdir=tmp_path, seed=42)
        
        manifest = result.manifest
        assert "metrics_per_fold" in manifest.artifacts
        assert "metrics_summary" in manifest.artifacts
        assert "best_params" in manifest.artifacts
        assert "metrics.csv" in manifest.artifacts["metrics"]
        assert "best_params.csv" in manifest.artifacts["best_params"]


class TestEndToEndIntegration:
    """End-to-end integration tests for artifact saving."""

    def test_full_workflow_standard_cv(self, tmp_path):
        """Test complete workflow: evaluate_model_cv -> save artifacts -> verify files."""
        from foodspec.validation.evaluation import evaluate_model_cv
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model = LogisticRegression(random_state=42, max_iter=100)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        result = evaluate_model_cv(
            X, y, model, splitter, metrics=["accuracy", "macro_f1"], seed=42
        )
        
        # Save artifacts
        result.save_predictions_csv(tmp_path / "predictions.csv")
        result.save_metrics_csv(tmp_path / "metrics.csv")
        
        # Verify files
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "metrics.csv").exists()
        
        pred_df = pd.read_csv(tmp_path / "predictions.csv")
        assert len(pred_df) == 50  # All samples predicted once
        assert set(pred_df.columns) >= {"fold_id", "sample_idx", "y_true", "y_pred"}
        
        metrics_df = pd.read_csv(tmp_path / "metrics.csv")
        # 5 folds + mean + std + ci_lower + ci_median + ci_upper = 10 rows
        assert len(metrics_df) == 10

    def test_full_workflow_nested_cv(self, tmp_path):
        """Test complete workflow: evaluate_model_nested_cv -> save artifacts -> verify files."""
        from foodspec.validation.evaluation import evaluate_model_nested_cv
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model_factory = lambda C=1.0: LogisticRegression(C=C, random_state=42, max_iter=100)
        outer_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        param_grid = {"C": [0.1, 1.0, 10.0]}
        
        result = evaluate_model_nested_cv(
            X, y, model_factory, outer_splitter, inner_splitter,
            param_grid=param_grid, metrics=["accuracy"], seed=42
        )
        
        # Save artifacts
        result.save_predictions_csv(tmp_path / "predictions.csv")
        result.save_metrics_csv(tmp_path / "metrics.csv")
        result.save_best_params_csv(tmp_path / "best_params.csv")
        
        # Verify files
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "metrics.csv").exists()
        assert (tmp_path / "best_params.csv").exists()
        
        params_df = pd.read_csv(tmp_path / "best_params.csv")
        assert len(params_df) == 3  # 3 outer folds
        assert "C" in params_df.columns
        assert all(params_df["C"].isin([0.1, 1.0, 10.0]))
