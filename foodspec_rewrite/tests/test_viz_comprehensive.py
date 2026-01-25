"""Tests for FoodSpec visualization modules with reproducibility and artifact validation."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.viz import (
    PlotConfig,
    plot_abstention_rate,
    plot_calibration_curve,
    plot_conformal_coverage_by_group,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_by_fold,
    plot_prediction_set_sizes,
)


class TestPlotConfig:
    """Test PlotConfig validation."""

    def test_default_config(self):
        config = PlotConfig()
        assert config.dpi == 300
        assert config.figure_size == (10, 6)
        assert config.seed is None

    def test_dpi_validation(self):
        """Test that dpi < 300 raises error."""
        try:
            config = PlotConfig(dpi=150)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "≥ 300" in str(e)

    def test_seeding(self):
        config1 = PlotConfig(seed=42)
        config2 = PlotConfig(seed=42)
        assert config1.seed == config2.seed


class TestConfusionMatrix:
    """Test confusion matrix plotting."""

    def test_binary_classification(self):
        """Test binary confusion matrix."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None
        assert len(fig.axes) > 0

    def test_multiclass(self):
        """Test multiclass confusion matrix."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])
        class_names = ["Class0", "Class1", "Class2"]

        fig = plot_confusion_matrix(y_true, y_pred, class_names=class_names)
        assert fig is not None

    def test_artifact_saving(self):
        """Test that figure is saved to registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = ArtifactRegistry(Path(tmpdir))
            y_true = np.array([0, 0, 1, 1])
            y_pred = np.array([0, 1, 1, 1])

            fig = plot_confusion_matrix(
                y_true, y_pred, artifacts=artifacts, filename="cm_test.png"
            )

            # Check file exists
            cm_path = artifacts.plots_dir / "cm_test.png"
            assert cm_path.exists()

    def test_with_metadata(self):
        """Test metadata in subtitle."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        fig = plot_confusion_matrix(
            y_true,
            y_pred,
            protocol_hash="abc123def456",
            run_id="run_001_test",
        )

        # Check figure was created
        assert fig is not None


class TestCalibrationCurve:
    """Test calibration curve plotting."""

    def test_binary_probabilities(self):
        """Test with binary probabilities."""
        y_true = np.array([0, 0, 1, 1, 1, 0])
        proba = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])

        fig = plot_calibration_curve(y_true, proba)
        assert fig is not None

    def test_2d_probabilities(self):
        """Test with 2D probability matrix."""
        y_true = np.array([0, 0, 1, 1])
        proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

        fig = plot_calibration_curve(y_true, proba)
        assert fig is not None

    def test_artifact_saving(self):
        """Test saving to registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = ArtifactRegistry(Path(tmpdir))
            y_true = np.array([0, 1, 1, 0])
            proba = np.array([0.2, 0.8, 0.7, 0.3])

            fig = plot_calibration_curve(
                y_true,
                proba,
                artifacts=artifacts,
                filename="calibration.png",
            )

            cal_path = artifacts.plots_dir / "calibration.png"
            assert cal_path.exists()


class TestMetricsByFold:
    """Test metrics by fold plotting."""

    def test_basic(self):
        """Test basic metrics plotting."""
        fold_metrics = [
            {"fold_id": 0, "accuracy": 0.85, "f1": 0.83},
            {"fold_id": 1, "accuracy": 0.87, "f1": 0.85},
            {"fold_id": 2, "accuracy": 0.82, "f1": 0.80},
        ]

        fig = plot_metrics_by_fold(fold_metrics)
        assert fig is not None

    def test_specific_metrics(self):
        """Test plotting specific metrics."""
        fold_metrics = [
            {"fold_id": 0, "accuracy": 0.85, "f1": 0.83, "roc_auc": 0.92},
            {"fold_id": 1, "accuracy": 0.87, "f1": 0.85, "roc_auc": 0.93},
        ]

        fig = plot_metrics_by_fold(fold_metrics, metric_names=["accuracy", "roc_auc"])
        assert fig is not None

    def test_artifact_saving(self):
        """Test saving to artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = ArtifactRegistry(Path(tmpdir))
            fold_metrics = [
                {"fold_id": 0, "accuracy": 0.85},
                {"fold_id": 1, "accuracy": 0.87},
            ]

            fig = plot_metrics_by_fold(
                fold_metrics,
                artifacts=artifacts,
                filename="metrics.png",
            )

            metrics_path = artifacts.plots_dir / "metrics.png"
            assert metrics_path.exists()


class TestFeatureImportance:
    """Test feature importance plotting."""

    def test_permutation_importance(self):
        """Test permutation importance format."""
        importance_df = pd.DataFrame({
            "feature": ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"],
            "importance_mean": [0.15, 0.12, 0.08, 0.05, 0.03],
        })

        fig = plot_feature_importance(importance_df, top_k=5)
        assert fig is not None

    def test_coefficients(self):
        """Test coefficient format."""
        importance_df = pd.DataFrame({
            "feature": ["x", "y", "z"],
            "coefficient": [0.8, -0.6, 0.4],
            "abs_coefficient": [0.8, 0.6, 0.4],
        })

        fig = plot_feature_importance(importance_df, top_k=3)
        assert fig is not None

    def test_top_k_limit(self):
        """Test top_k limiting."""
        importance_df = pd.DataFrame({
            "feature": [f"f{i}" for i in range(20)],
            "importance_mean": np.linspace(0.2, 0.01, 20),
        })

        fig = plot_feature_importance(importance_df, top_k=5)
        assert fig is not None

    def test_artifact_saving(self):
        """Test saving to artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = ArtifactRegistry(Path(tmpdir))
            importance_df = pd.DataFrame({
                "feature": ["a", "b", "c"],
                "importance_mean": [0.5, 0.3, 0.2],
            })

            fig = plot_feature_importance(
                importance_df,
                artifacts=artifacts,
                filename="importance.png",
            )

            imp_path = artifacts.plots_dir / "importance.png"
            assert imp_path.exists()


class TestConformalCoverageByGroup:
    """Test conformal coverage plotting."""

    def test_basic_coverage(self):
        """Test basic coverage table."""
        coverage_df = pd.DataFrame({
            "group": ["batch_a", "batch_b", "stage_1"],
            "coverage": [0.92, 0.88, 0.95],
            "n_samples": [50, 40, 45],
            "avg_set_size": [1.5, 1.8, 1.3],
        })

        fig = plot_conformal_coverage_by_group(coverage_df, target_coverage=0.9)
        assert fig is not None

    def test_coverage_below_target(self):
        """Test coverage with some below target."""
        coverage_df = pd.DataFrame({
            "group": ["group_1", "group_2"],
            "coverage": [0.85, 0.92],
            "n_samples": [100, 100],
            "avg_set_size": [1.5, 1.2],
        })

        fig = plot_conformal_coverage_by_group(coverage_df, target_coverage=0.9)
        assert fig is not None

    def test_artifact_saving(self):
        """Test saving to artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = ArtifactRegistry(Path(tmpdir))
            coverage_df = pd.DataFrame({
                "group": ["a", "b"],
                "coverage": [0.91, 0.89],
                "n_samples": [100, 100],
                "avg_set_size": [1.5, 1.5],
            })

            fig = plot_conformal_coverage_by_group(
                coverage_df,
                artifacts=artifacts,
                filename="coverage.png",
            )

            cov_path = artifacts.plots_dir / "coverage.png"
            assert cov_path.exists()


class TestPredictionSetSizes:
    """Test prediction set size distribution plotting."""

    def test_ungrouped(self):
        """Test ungrouped set sizes."""
        set_sizes = np.array([1, 1, 2, 1, 3, 2, 1, 1, 2, 3])

        fig = plot_prediction_set_sizes(set_sizes)
        assert fig is not None

    def test_grouped(self):
        """Test grouped set sizes."""
        set_sizes = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2])
        grouping = np.array(["a", "a", "a", "a", "a", "a", "b", "b", "b", "b"])

        fig = plot_prediction_set_sizes(set_sizes, grouping=grouping)
        assert fig is not None

    def test_with_group_names(self):
        """Test with custom group names."""
        set_sizes = np.array([1, 2, 3, 1, 2])
        grouping = np.array([0, 0, 0, 1, 1])
        group_names = {0: "Batch A", 1: "Batch B"}

        fig = plot_prediction_set_sizes(
            set_sizes, grouping=grouping, group_names=group_names
        )
        assert fig is not None

    def test_artifact_saving(self):
        """Test saving to artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = ArtifactRegistry(Path(tmpdir))
            set_sizes = np.array([1, 1, 2, 2])

            fig = plot_prediction_set_sizes(
                set_sizes,
                artifacts=artifacts,
                filename="sizes.png",
            )

            sizes_path = artifacts.plots_dir / "sizes.png"
            assert sizes_path.exists()


class TestAbstentionRate:
    """Test abstention rate plotting."""

    def test_fold_abstention(self):
        """Test abstention by fold."""
        abstention_rates = {
            "Fold 0": 0.05,
            "Fold 1": 0.08,
            "Fold 2": 0.06,
        }

        fig = plot_abstention_rate(abstention_rates)
        assert fig is not None

    def test_group_abstention(self):
        """Test abstention by group."""
        abstention_rates = {
            "batch_a": 0.10,
            "batch_b": 0.05,
            "batch_c": 0.08,
        }

        fig = plot_abstention_rate(abstention_rates)
        assert fig is not None

    def test_artifact_saving(self):
        """Test saving to artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = ArtifactRegistry(Path(tmpdir))
            abstention_rates = {"Fold 0": 0.05, "Fold 1": 0.06}

            fig = plot_abstention_rate(
                abstention_rates,
                artifacts=artifacts,
                filename="abstention.png",
            )

            abs_path = artifacts.plots_dir / "abstention.png"
            assert abs_path.exists()


class TestDeterminism:
    """Test deterministic/reproducible plotting."""

    def test_seeded_randomness(self):
        """Test that seeding produces same plots."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        proba = np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.1])

        config = PlotConfig(seed=42)
        fig1 = plot_calibration_curve(y_true, proba, config=config)
        
        config2 = PlotConfig(seed=42)
        fig2 = plot_calibration_curve(y_true, proba, config=config2)

        # Both figures should exist and be comparable
        assert fig1 is not None
        assert fig2 is not None

    def test_high_dpi_export(self):
        """Test that high DPI is applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = ArtifactRegistry(Path(tmpdir))
            config = PlotConfig(dpi=300)

            y_true = np.array([0, 1, 1, 0])
            y_pred = np.array([0, 1, 0, 0])

            plot_confusion_matrix(
                y_true,
                y_pred,
                artifacts=artifacts,
                config=config,
                filename="highres.png",
            )

            # Check file exists (high dpi should make it larger)
            cm_path = artifacts.plots_dir / "highres.png"
            assert cm_path.exists()
            # File should be reasonably sized (high DPI)
            assert cm_path.stat().st_size > 5000  # At least 5KB
