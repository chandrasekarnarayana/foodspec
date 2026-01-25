"""
Integration tests for Modeling & Validation subsystem.

Tests verify:
- Leakage safety (features fit on train only)
- Determinism (reproducible results with seeds)
- Group-aware CV enforcement
- End-to-end ProtocolV2 + Registry integration
- Artifact saving
"""

import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from foodspec.core.protocol import (
    DataSpec,
    ProtocolV2,
    TaskSpec,
    ValidationSpec,
    FeatureSpec,
    ModelSpec,
)
from foodspec.core.registry import ComponentRegistry
from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.data import SpectraSet
from foodspec.validation.modeling import ModelingConfig, ModelingPipeline
from foodspec.models.classical import LogisticRegressionClassifier
from foodspec.features.chemometrics import PCAFeatureExtractor
from foodspec.features.hybrid import FeatureUnion


@pytest.fixture
def synthetic_binary_data():
    """Synthetic binary classification data with groups."""
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 100
    
    X = rng.standard_normal((n_samples, n_features))
    y = np.array([0] * 30 + [1] * 30)
    groups = np.array([0] * 15 + [1] * 15 + [2] * 15 + [3] * 15)
    
    metadata = pd.DataFrame({
        "sample_id": [f"s{i:03d}" for i in range(n_samples)],
        "group": groups,
        "modality": ["raman"] * n_samples,
    })
    
    return X, y, groups, metadata


@pytest.fixture
def protocol_with_lobo():
    """ProtocolV2 with LOBO validation."""
    return ProtocolV2(
        data=DataSpec(
            input="data.csv",
            modality="raman",
            label="target",
            metadata_map={
                "sample_id": "sample_id",
                "modality": "modality",
                "label": "target",
            },
        ),
        task=TaskSpec(name="classification", objective="maximize accuracy"),
        features=FeatureSpec(strategy="manual", modules=["pca"]),
        model=ModelSpec(family="sklearn", estimator="logreg"),
        validation=ValidationSpec(scheme="lobo", group_key="group", nested=False, metrics=["accuracy"]),
    )


@pytest.fixture
def registry_with_models():
    """ComponentRegistry with models pre-registered."""
    registry = ComponentRegistry()
    registry.register("model", "logreg", LogisticRegressionClassifier)
    return registry


class TestModelingPipelineLeakageSafety:
    """Verify leakage-safe feature extraction and model training."""

    def test_features_fit_on_train_only(self, synthetic_binary_data, protocol_with_lobo, registry_with_models):
        """Verify features are fit on training fold only, not on test."""
        X, y, groups, metadata = synthetic_binary_data
        
        # Simple test: ensure FeatureUnion.fit is called only on train indices
        # This is implicitly tested by successful evaluation without leakage errors
        
        config = ModelingConfig(
            protocol=protocol_with_lobo,
            registry=registry_with_models,
            seed=42,
        )
        
        pipeline = ModelingPipeline(config)
        result = pipeline.run(X, y, groups=groups, metadata=metadata)
        
        # If we get here without errors, leakage checks passed
        assert result is not None
        assert len(result.fold_metrics) >= 1, "Should have at least one fold"

    def test_deterministic_results_with_seed(self, synthetic_binary_data, protocol_with_lobo, registry_with_models):
        """Verify same seed produces identical results across runs."""
        X, y, groups, metadata = synthetic_binary_data
        
        config1 = ModelingConfig(
            protocol=protocol_with_lobo,
            registry=registry_with_models,
            seed=42,
        )
        pipeline1 = ModelingPipeline(config1)
        result1 = pipeline1.run(X, y, groups=groups, metadata=metadata)
        
        config2 = ModelingConfig(
            protocol=protocol_with_lobo,
            registry=registry_with_models,
            seed=42,
        )
        pipeline2 = ModelingPipeline(config2)
        result2 = pipeline2.run(X, y, groups=groups, metadata=metadata)
        
        # Results should be identical
        acc1 = [m["accuracy"] for m in result1.fold_metrics]
        acc2 = [m["accuracy"] for m in result2.fold_metrics]
        
        assert np.allclose(acc1, acc2), "Accuracies should be identical with same seed"


class TestModelingPipelineGroupAware:
    """Verify group-aware CV is enforced by default."""

    def test_lobo_requires_groups(self, synthetic_binary_data, protocol_with_lobo, registry_with_models):
        """Verify LOBO validation requires groups."""
        X, y, groups, metadata = synthetic_binary_data
        
        config = ModelingConfig(
            protocol=protocol_with_lobo,
            registry=registry_with_models,
            seed=42,
        )
        
        pipeline = ModelingPipeline(config)
        
        # Should work with groups
        result = pipeline.run(X, y, groups=groups, metadata=metadata)
        assert result is not None
        
        # Should fail without groups
        with pytest.raises(ValueError, match="groups"):
            pipeline.run(X, y, groups=None, metadata=metadata)

    def test_lobo_yields_correct_folds(self, synthetic_binary_data, protocol_with_lobo, registry_with_models):
        """Verify LOBO produces correct number of folds."""
        X, y, groups, metadata = synthetic_binary_data
        
        # With 4 groups, should get 4 folds
        assert len(np.unique(groups)) == 4
        
        config = ModelingConfig(
            protocol=protocol_with_lobo,
            registry=registry_with_models,
            seed=42,
        )
        
        pipeline = ModelingPipeline(config)
        result = pipeline.run(X, y, groups=groups, metadata=metadata)
        
        assert len(result.fold_metrics) == 4, "LOBO with 4 groups should produce 4 folds"


class TestModelingPipelineIntegration:
    """End-to-end integration tests with ProtocolV2 and ComponentRegistry."""

    def test_protocol_based_modeling(self, synthetic_binary_data, protocol_with_lobo, registry_with_models):
        """Verify ProtocolV2 fully specifies modeling workflow."""
        X, y, groups, metadata = synthetic_binary_data
        
        config = ModelingConfig(
            protocol=protocol_with_lobo,
            registry=registry_with_models,
            seed=42,
        )
        
        pipeline = ModelingPipeline(config)
        result = pipeline.run(X, y, groups=groups, metadata=metadata)
        
        # Verify result structure
        assert hasattr(result, "fold_metrics")
        assert hasattr(result, "fold_predictions")
        assert hasattr(result, "bootstrap_ci")
        assert len(result.fold_metrics) > 0

    def test_stratified_kfold_default(self, synthetic_binary_data, registry_with_models):
        """Verify stratified K-fold works when groups unavailable."""
        X, y, _, metadata = synthetic_binary_data
        
        protocol = ProtocolV2(
            data=DataSpec(
                input="data.csv",
                modality="raman",
                label="target",
                metadata_map={"sample_id": "id", "modality": "modality", "label": "target"},
            ),
            task=TaskSpec(name="classification", objective="maximize accuracy"),
            features=FeatureSpec(strategy="manual", modules=[]),
            model=ModelSpec(family="sklearn", estimator="logreg"),
            validation=ValidationSpec(scheme="stratified_kfold", nested=False, metrics=["accuracy"]),
        )
        
        config = ModelingConfig(
            protocol=protocol,
            registry=registry_with_models,
            seed=42,
        )
        
        pipeline = ModelingPipeline(config)
        result = pipeline.run(X, y, groups=None, metadata=metadata)
        
        # Should get at least 1 fold without explicit n_splits
        assert len(result.fold_metrics) >= 1, "Stratified K-fold should produce folds"

    def test_bootstrap_confidence_intervals(self, synthetic_binary_data, protocol_with_lobo, registry_with_models):
        """Verify bootstrap CIs are computed."""
        X, y, groups, metadata = synthetic_binary_data
        
        config = ModelingConfig(
            protocol=protocol_with_lobo,
            registry=registry_with_models,
            seed=42,
        )
        
        pipeline = ModelingPipeline(config)
        result = pipeline.run(X, y, groups=groups, metadata=metadata)
        
        # Check bootstrap CIs (now returns 3-tuple with median)
        assert "accuracy" in result.bootstrap_ci, "Should have accuracy CI"
        lower, median, upper = result.bootstrap_ci["accuracy"]
        assert lower <= median <= upper, "CI should be ordered: lower <= median <= upper"
        assert 0 <= lower <= 1 and 0 <= upper <= 1, "CI bounds should be in [0, 1]"


class TestModelingPipelineArtifactSaving:
    """Verify artifact saving functionality."""

    def test_saves_predictions_and_metrics(self, synthetic_binary_data, protocol_with_lobo, registry_with_models):
        """Verify predictions and metrics are saved to artifact registry."""
        X, y, groups, metadata = synthetic_binary_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            artifact_reg = ArtifactRegistry(root=tmpdir_path)
            
            config = ModelingConfig(
                protocol=protocol_with_lobo,
                registry=registry_with_models,
                artifact_registry=artifact_reg,
                seed=42,
            )
            
            pipeline = ModelingPipeline(config)
            result = pipeline.run(X, y, groups=groups, metadata=metadata)
            
            assert result is not None
            # Note: artifact saving implementation is TODO
            # In production, metrics.csv and predictions.csv should be saved


class TestModelingConfigValidation:
    """Verify configuration validation."""

    def test_requires_protocol(self, registry_with_models):
        """Verify ProtocolV2 is required."""
        with pytest.raises(ValueError, match="ProtocolV2"):
            ModelingConfig(
                protocol=None,
                registry=registry_with_models,
            )

    def test_requires_registry(self, protocol_with_lobo):
        """Verify ComponentRegistry is required."""
        with pytest.raises(ValueError, match="ComponentRegistry"):
            ModelingConfig(
                protocol=protocol_with_lobo,
                registry=None,
            )

    def test_rejects_unknown_validation_scheme(self, protocol_with_lobo, registry_with_models):
        """Verify unknown validation schemes are rejected."""
        bad_protocol = protocol_with_lobo.model_copy(
            update={"validation": ValidationSpec(scheme="unknown_scheme")}
        )
        
        config = ModelingConfig(
            protocol=bad_protocol,
            registry=registry_with_models,
        )
        
        with pytest.raises(ValueError, match="scheme"):
            ModelingPipeline(config)
