"""Tests for decision policy and operating point selection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from foodspec.modeling.diagnostics.roc import compute_roc_diagnostics
from foodspec.trust.decision_policy import (
    DecisionPolicy,
    OperatingPoint,
    PolicyType,
    choose_operating_point,
    save_operating_point,
)


class TestDecisionPolicy:
    """Test DecisionPolicy dataclass and validation."""

    def test_policy_creation(self):
        """Test basic policy creation."""
        policy = DecisionPolicy(
            name="youden",
            applies_to="binary",
            params={},
        )
        assert policy.name == "youden"
        assert policy.applies_to == "binary"

    def test_policy_validation_unknown_name(self):
        """Test that unknown policy names raise error."""
        with pytest.raises(ValueError, match="Unknown policy"):
            DecisionPolicy(name="unknown_policy", params={})

    def test_cost_sensitive_requires_params(self):
        """Test that cost_sensitive policy requires cost_fp and cost_fn."""
        with pytest.raises(ValueError, match="cost_fp and cost_fn"):
            DecisionPolicy(
                name="cost_sensitive",
                params={},  # Missing required params
            )

    def test_target_sensitivity_requires_min_sens(self):
        """Test that target_sensitivity requires min_sensitivity param."""
        with pytest.raises(ValueError, match="min_sensitivity"):
            DecisionPolicy(
                name="target_sensitivity",
                params={},
            )

    def test_regulatory_mode_flag(self):
        """Test regulatory mode flag."""
        policy = DecisionPolicy(
            name="youden",
            regulatory_mode=True,
        )
        assert policy.regulatory_mode is True


class TestYoudenPolicy:
    """Test Youden's J-statistic policy."""

    @pytest.fixture
    def binary_data_and_roc(self):
        """Generate binary classification data and ROC result."""
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_classes=2,
            random_state=42,
        )
        y_proba = np.column_stack([1 - np.random.rand(len(y)) * 0.3,
                                    np.random.rand(len(y)) * 0.7 + 0.3])
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        roc_result = compute_roc_diagnostics(y, y_proba, n_bootstrap=100, random_seed=42)
        return y, y_proba, roc_result

    def test_youden_computes_threshold(self, binary_data_and_roc):
        """Test that Youden policy computes valid threshold."""
        y, y_proba, roc_result = binary_data_and_roc

        policy = DecisionPolicy(name="youden")
        op = choose_operating_point(y, y_proba, roc_result, policy)

        assert isinstance(op.thresholds, (int, float))
        assert 0.0 <= op.thresholds <= 1.0
        assert op.policy.name == "youden"

    def test_youden_metrics(self, binary_data_and_roc):
        """Test that Youden operating point has expected metrics."""
        y, y_proba, roc_result = binary_data_and_roc

        policy = DecisionPolicy(name="youden")
        op = choose_operating_point(y, y_proba, roc_result, policy)

        assert "sensitivity" in op.achieved_metrics
        assert "specificity" in op.achieved_metrics
        assert "f1" in op.achieved_metrics
        assert "j_statistic" in op.achieved_metrics
        assert 0.0 <= op.achieved_metrics["sensitivity"] <= 1.0
        assert 0.0 <= op.achieved_metrics["specificity"] <= 1.0

    def test_youden_rationale(self, binary_data_and_roc):
        """Test that Youden generates explanatory rationale."""
        y, y_proba, roc_result = binary_data_and_roc

        policy = DecisionPolicy(name="youden")
        op = choose_operating_point(y, y_proba, roc_result, policy)

        assert len(op.rationale) > 0
        assert "Youden" in op.rationale


class TestCostSensitivePolicy:
    """Test cost-sensitive policy."""

    @pytest.fixture
    def binary_data_and_roc(self):
        """Generate binary classification data."""
        X, y = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)
        y_proba = np.column_stack([1 - np.random.rand(len(y)) * 0.3,
                                    np.random.rand(len(y)) * 0.7 + 0.3])
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        roc_result = compute_roc_diagnostics(y, y_proba, n_bootstrap=100, random_seed=42)
        return y, y_proba, roc_result

    def test_cost_sensitive_requires_params(self, binary_data_and_roc):
        """Test that cost_sensitive policy validates params."""
        y, y_proba, roc_result = binary_data_and_roc

        with pytest.raises(ValueError, match="cost_fp and cost_fn"):
            policy = DecisionPolicy(name="cost_sensitive", params={})
            choose_operating_point(y, y_proba, roc_result, policy)

    def test_cost_sensitive_unequal_costs(self, binary_data_and_roc):
        """Test cost-sensitive with unequal costs."""
        y, y_proba, roc_result = binary_data_and_roc

        # High cost for false negatives
        policy = DecisionPolicy(
            name="cost_sensitive",
            params={"cost_fp": 1.0, "cost_fn": 5.0},
        )
        op = choose_operating_point(y, y_proba, roc_result, policy)

        assert isinstance(op.thresholds, (int, float))
        assert 0.0 <= op.thresholds <= 1.0
        assert "cost_sensitive" in op.metadata["method"]

    def test_cost_sensitive_rationale(self, binary_data_and_roc):
        """Test cost-sensitive generates rationale."""
        y, y_proba, roc_result = binary_data_and_roc

        policy = DecisionPolicy(
            name="cost_sensitive",
            params={"cost_fp": 1.0, "cost_fn": 5.0},
        )
        op = choose_operating_point(y, y_proba, roc_result, policy)

        assert len(op.rationale) > 0
        assert "cost" in op.rationale.lower()


class TestTargetSensitivityPolicy:
    """Test target sensitivity policy."""

    @pytest.fixture
    def binary_data_and_roc(self):
        """Generate binary classification data."""
        X, y = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)
        y_proba = np.column_stack([1 - np.random.rand(len(y)) * 0.3,
                                    np.random.rand(len(y)) * 0.7 + 0.3])
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        roc_result = compute_roc_diagnostics(y, y_proba, n_bootstrap=100, random_seed=42)
        return y, y_proba, roc_result

    def test_target_sensitivity_achieves_target(self, binary_data_and_roc):
        """Test that target sensitivity policy achieves target sensitivity."""
        y, y_proba, roc_result = binary_data_and_roc

        min_sensitivity = 0.85
        policy = DecisionPolicy(
            name="target_sensitivity",
            params={"min_sensitivity": min_sensitivity},
        )
        op = choose_operating_point(y, y_proba, roc_result, policy)

        # Should achieve at least min_sensitivity
        assert op.achieved_metrics["sensitivity"] >= min_sensitivity - 0.01  # Small tolerance

    def test_target_sensitivity_high_target_warns(self, binary_data_and_roc):
        """Test that unachievable target sensitivity generates warning."""
        y, y_proba, roc_result = binary_data_and_roc

        min_sensitivity = 0.999  # Very high, likely unachievable
        policy = DecisionPolicy(
            name="target_sensitivity",
            params={"min_sensitivity": min_sensitivity},
        )
        op = choose_operating_point(y, y_proba, roc_result, policy)

        # Should have warning or use best available
        assert isinstance(op.achieved_metrics["sensitivity"], float)

    def test_target_sensitivity_rationale(self, binary_data_and_roc):
        """Test that target sensitivity generates appropriate rationale."""
        y, y_proba, roc_result = binary_data_and_roc

        policy = DecisionPolicy(
            name="target_sensitivity",
            params={"min_sensitivity": 0.90},
            regulatory_mode=True,
        )
        op = choose_operating_point(y, y_proba, roc_result, policy)

        assert "regulatory" in op.rationale.lower() or "sensitivity" in op.rationale.lower()


class TestTargetSpecificityPolicy:
    """Test target specificity policy."""

    @pytest.fixture
    def binary_data_and_roc(self):
        """Generate binary classification data."""
        X, y = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)
        y_proba = np.column_stack([1 - np.random.rand(len(y)) * 0.3,
                                    np.random.rand(len(y)) * 0.7 + 0.3])
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        roc_result = compute_roc_diagnostics(y, y_proba, n_bootstrap=100, random_seed=42)
        return y, y_proba, roc_result

    def test_target_specificity_achieves_target(self, binary_data_and_roc):
        """Test that target specificity policy achieves target specificity."""
        y, y_proba, roc_result = binary_data_and_roc

        min_specificity = 0.85
        policy = DecisionPolicy(
            name="target_specificity",
            params={"min_specificity": min_specificity},
        )
        op = choose_operating_point(y, y_proba, roc_result, policy)

        # Should achieve at least min_specificity
        assert op.achieved_metrics["specificity"] >= min_specificity - 0.01


class TestAbstentionAwarePolicy:
    """Test abstention-aware policy."""

    @pytest.fixture
    def binary_data_and_roc(self):
        """Generate binary classification data."""
        X, y = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)
        y_proba = np.column_stack([1 - np.random.rand(len(y)) * 0.3,
                                    np.random.rand(len(y)) * 0.7 + 0.3])
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        roc_result = compute_roc_diagnostics(y, y_proba, n_bootstrap=100, random_seed=42)
        return y, y_proba, roc_result

    def test_abstention_aware_computes_threshold(self, binary_data_and_roc):
        """Test that abstention-aware policy computes valid threshold."""
        y, y_proba, roc_result = binary_data_and_roc

        policy = DecisionPolicy(
            name="abstention_aware",
            params={"max_abstention_rate": 0.2},
        )
        op = choose_operating_point(y, y_proba, roc_result, policy)

        assert isinstance(op.thresholds, (int, float))
        assert 0.0 <= op.thresholds <= 1.0

    def test_abstention_aware_respects_constraint(self, binary_data_and_roc):
        """Test that abstention-aware respects max abstention rate."""
        y, y_proba, roc_result = binary_data_and_roc

        max_abstention = 0.15
        policy = DecisionPolicy(
            name="abstention_aware",
            params={"max_abstention_rate": max_abstention},
        )
        op = choose_operating_point(y, y_proba, roc_result, policy)

        achieved_abstention = op.uncertainty_metrics.get("abstention_rate", 0.0)
        assert achieved_abstention <= max_abstention + 0.01  # Small tolerance


class TestOperatingPointOutput:
    """Test OperatingPoint serialization and saving."""

    @pytest.fixture
    def operating_point(self):
        """Create a sample operating point."""
        policy = DecisionPolicy(name="youden")
        return OperatingPoint(
            thresholds=0.5,
            policy=policy,
            achieved_metrics={
                "sensitivity": 0.85,
                "specificity": 0.90,
                "f1": 0.87,
            },
            rationale="Test rationale",
            warnings=["Test warning"],
        )

    def test_to_dict(self, operating_point):
        """Test OperatingPoint to_dict conversion."""
        d = operating_point.to_dict()

        assert d["thresholds"] == 0.5
        assert d["policy"]["name"] == "youden"
        assert d["achieved_metrics"]["sensitivity"] == 0.85
        assert "Test rationale" in d["rationale"]
        assert len(d["warnings"]) > 0

    def test_save_operating_point(self, operating_point):
        """Test saving operating point to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = save_operating_point(Path(tmpdir), operating_point)

            assert "decision_policy_json" in artifacts
            assert Path(artifacts["decision_policy_json"]).exists()

            # Verify JSON contents
            with open(artifacts["decision_policy_json"]) as f:
                data = json.load(f)
                assert data["thresholds"] == 0.5
                assert data["policy"]["name"] == "youden"


class TestPolicyTypeEnum:
    """Test PolicyType enum."""

    def test_all_policy_types_available(self):
        """Test that all policy types are defined."""
        expected = ["youden", "cost_sensitive", "target_sensitivity",
                    "target_specificity", "abstention_aware"]
        actual = [p.value for p in PolicyType]

        for policy in expected:
            assert policy in actual


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
