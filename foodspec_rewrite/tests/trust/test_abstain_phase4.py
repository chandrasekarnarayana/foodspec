"""
Tests for abstention policies (Phase 4).

Tests:
  - MaxProbAbstainer: threshold-based abstention
  - ConformalSizeAbstainer: set-size-based abstention
  - CombinedAbstainer: combining rules with any/all modes
  - Metrics: abstain_rate, accuracy_on_answered computed correctly
  - Reason codes: stable and meaningful
"""

import numpy as np
import pytest

from foodspec.trust.abstain import (
    CombinedAbstainer,
    ConformalSizeAbstainer,
    MaxProbAbstainer,
)


class TestMaxProbAbstainer:
    """Test MaxProbAbstainer."""

    def test_init_valid_threshold(self):
        """Test initialization with valid threshold."""
        abstainer = MaxProbAbstainer(threshold=0.5)
        assert abstainer.threshold == 0.5

    def test_init_invalid_threshold_too_low(self):
        """Test initialization with threshold too low."""
        with pytest.raises(ValueError, match="threshold must be in"):
            MaxProbAbstainer(threshold=0.0)

    def test_init_invalid_threshold_too_high(self):
        """Test initialization with threshold too high."""
        with pytest.raises(ValueError, match="threshold must be in"):
            MaxProbAbstainer(threshold=1.1)

    def test_apply_basic(self):
        """Test basic apply functionality."""
        abstainer = MaxProbAbstainer(threshold=0.6)
        proba = np.array([[0.7, 0.3], [0.55, 0.45], [0.9, 0.1]])

        mask, reasons = abstainer.apply(proba)

        assert mask.dtype == bool
        assert len(mask) == 3
        assert mask[0] == False  # 0.7 > 0.6
        assert mask[1] == True  # 0.55 <= 0.6
        assert mask[2] == False  # 0.9 > 0.6

    def test_apply_reason_codes(self):
        """Test reason codes are generated correctly."""
        abstainer = MaxProbAbstainer(threshold=0.6)
        proba = np.array([[0.7, 0.3], [0.55, 0.45], [0.9, 0.1]])

        mask, reasons = abstainer.apply(proba)

        assert reasons == ["confident", "low_confidence", "confident"]

    def test_apply_all_abstain(self):
        """Test when all samples abstain."""
        abstainer = MaxProbAbstainer(threshold=0.9)
        proba = np.array([[0.5, 0.5], [0.6, 0.4]])

        mask, reasons = abstainer.apply(proba)

        assert mask.all()
        assert all(r == "low_confidence" for r in reasons)

    def test_apply_none_abstain(self):
        """Test when no samples abstain."""
        abstainer = MaxProbAbstainer(threshold=0.1)
        proba = np.array([[0.9, 0.1], [0.8, 0.2]])

        mask, reasons = abstainer.apply(proba)

        assert not mask.any()
        assert all(r == "confident" for r in reasons)

    def test_apply_boundary_threshold(self):
        """Test boundary at threshold value."""
        abstainer = MaxProbAbstainer(threshold=0.5)
        proba = np.array([[0.5, 0.5], [0.50001, 0.49999]])

        mask, reasons = abstainer.apply(proba)

        assert mask[0] == True  # 0.5 <= 0.5
        assert mask[1] == False  # 0.50001 > 0.5

    def test_apply_invalid_proba_shape(self):
        """Test error on invalid proba shape."""
        abstainer = MaxProbAbstainer(threshold=0.6)
        proba_1d = np.array([0.7, 0.3])

        with pytest.raises(ValueError, match="proba must be 2D"):
            abstainer.apply(proba_1d)

    def test_evaluate_basic(self):
        """Test evaluate metrics."""
        abstainer = MaxProbAbstainer(threshold=0.6)
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        mask_abstain = np.array([False, True, False])

        metrics = abstainer.evaluate(y_true, y_pred, mask_abstain)

        assert metrics["abstain_rate"] == 1 / 3
        assert metrics["accuracy_on_answered"] == 1.0
        assert metrics["coverage_on_answered"] is None

    def test_evaluate_all_abstain(self):
        """Test evaluate when all samples abstain."""
        abstainer = MaxProbAbstainer(threshold=0.6)
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        mask_abstain = np.array([True, True])

        metrics = abstainer.evaluate(y_true, y_pred, mask_abstain)

        assert metrics["abstain_rate"] == 1.0
        assert metrics["accuracy_on_answered"] is None

    def test_evaluate_accuracy_correct(self):
        """Test accuracy_on_answered computation."""
        abstainer = MaxProbAbstainer(threshold=0.6)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])  # 3/4 correct
        mask_abstain = np.array([False, False, False, False])

        metrics = abstainer.evaluate(y_true, y_pred, mask_abstain)

        assert metrics["accuracy_on_answered"] == 0.75

    def test_evaluate_with_abstention(self):
        """Test accuracy_on_answered excluding abstained samples."""
        abstainer = MaxProbAbstainer(threshold=0.6)
        y_true = np.array([0, 1, 0])
        y_pred = np.array([1, 1, 0])  # 1st wrong, 2nd right (abstained), 3rd right
        mask_abstain = np.array([False, True, False])

        metrics = abstainer.evaluate(y_true, y_pred, mask_abstain)

        # Answered samples: 1st (wrong), 3rd (right) -> 0.5 accuracy
        assert metrics["accuracy_on_answered"] == 0.5

    def test_evaluate_length_mismatch(self):
        """Test error on length mismatch."""
        abstainer = MaxProbAbstainer(threshold=0.6)
        y_true = np.array([0, 1])
        y_pred = np.array([0])
        mask_abstain = np.array([False, True])

        with pytest.raises(ValueError, match="same length"):
            abstainer.evaluate(y_true, y_pred, mask_abstain)


class TestConformalSizeAbstainer:
    """Test ConformalSizeAbstainer."""

    def test_init_valid_size(self):
        """Test initialization with valid max_set_size."""
        abstainer = ConformalSizeAbstainer(max_set_size=2)
        assert abstainer.max_set_size == 2

    def test_init_invalid_size_zero(self):
        """Test initialization with invalid size."""
        with pytest.raises(ValueError, match="max_set_size must be positive"):
            ConformalSizeAbstainer(max_set_size=0)

    def test_init_invalid_size_negative(self):
        """Test initialization with negative size."""
        with pytest.raises(ValueError, match="max_set_size must be positive"):
            ConformalSizeAbstainer(max_set_size=-1)

    def test_apply_basic(self):
        """Test basic apply functionality."""
        abstainer = ConformalSizeAbstainer(max_set_size=2)
        conformal_sets = [[0, 1], [0, 1, 2], [1]]

        mask, reasons = abstainer.apply(None, conformal_sets)

        assert mask[0] == False  # size 2 <= max_set_size 2
        assert mask[1] == True  # size 3 > max_set_size 2
        assert mask[2] == False  # size 1 <= max_set_size 2

    def test_apply_reason_codes(self):
        """Test reason codes."""
        abstainer = ConformalSizeAbstainer(max_set_size=2)
        conformal_sets = [[0, 1], [0, 1, 2], [1]]

        mask, reasons = abstainer.apply(None, conformal_sets)

        assert reasons == ["small_set", "large_set", "small_set"]

    def test_apply_all_small(self):
        """Test when all sets are small."""
        abstainer = ConformalSizeAbstainer(max_set_size=5)
        conformal_sets = [[0], [1, 2], [0, 1, 2, 3]]

        mask, reasons = abstainer.apply(None, conformal_sets)

        assert not mask.any()
        assert all(r == "small_set" for r in reasons)

    def test_apply_all_large(self):
        """Test when all sets are large."""
        abstainer = ConformalSizeAbstainer(max_set_size=2)
        conformal_sets = [[0, 1, 2], [1, 2, 3], [0, 1, 2, 3]]

        mask, reasons = abstainer.apply(None, conformal_sets)

        assert mask.all()
        assert all(r == "large_set" for r in reasons)

    def test_apply_none_conformal_sets(self):
        """Test error when conformal_sets is None."""
        abstainer = ConformalSizeAbstainer(max_set_size=2)

        with pytest.raises(ValueError, match="conformal_sets cannot be None"):
            abstainer.apply(None, None)

    def test_evaluate_basic(self):
        """Test evaluate metrics."""
        abstainer = ConformalSizeAbstainer(max_set_size=2)
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        mask_abstain = np.array([False, True, False])

        metrics = abstainer.evaluate(y_true, y_pred, mask_abstain)

        assert metrics["abstain_rate"] == 1 / 3
        assert metrics["accuracy_on_answered"] == 1.0

    def test_evaluate_accuracy(self):
        """Test accuracy_on_answered computation."""
        abstainer = ConformalSizeAbstainer(max_set_size=2)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        mask_abstain = np.array([False, False, False, False])

        metrics = abstainer.evaluate(y_true, y_pred, mask_abstain)

        assert metrics["accuracy_on_answered"] == 0.75


class TestCombinedAbstainer:
    """Test CombinedAbstainer."""

    def test_init_empty_rules(self):
        """Test error on empty rules."""
        with pytest.raises(ValueError, match="rules cannot be empty"):
            CombinedAbstainer(rules=[])

    def test_init_invalid_mode(self):
        """Test error on invalid mode."""
        rule = MaxProbAbstainer(threshold=0.6)
        with pytest.raises(ValueError, match="mode must be"):
            CombinedAbstainer(rules=[rule], mode="invalid")

    def test_init_valid(self):
        """Test valid initialization."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        rule2 = ConformalSizeAbstainer(max_set_size=2)
        combined = CombinedAbstainer(rules=[rule1, rule2], mode="any")

        assert len(combined.rules) == 2
        assert combined.mode == "any"

    def test_apply_mode_any_single_rule(self):
        """Test mode='any' with single rule."""
        rule = MaxProbAbstainer(threshold=0.6)
        combined = CombinedAbstainer(rules=[rule], mode="any")
        proba = np.array([[0.7, 0.3], [0.55, 0.45]])

        mask, reasons = combined.apply(proba, None)

        assert mask[0] == False
        assert mask[1] == True

    def test_apply_mode_any_multiple_rules(self):
        """Test mode='any' with multiple rules."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        rule2 = ConformalSizeAbstainer(max_set_size=1)
        combined = CombinedAbstainer(rules=[rule1, rule2], mode="any")

        proba = np.array([[0.7, 0.3], [0.55, 0.45], [0.9, 0.1]])
        conformal_sets = [[0], [0, 1], [0]]

        mask, reasons = combined.apply(proba, conformal_sets)

        # Sample 0: prob OK (False), set small (False) -> False
        # Sample 1: prob low (True), set large (True) -> True
        # Sample 2: prob OK (False), set small (False) -> False
        assert mask[0] == False
        assert mask[1] == True
        assert mask[2] == False

    def test_apply_mode_all_multiple_rules(self):
        """Test mode='all' with multiple rules."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        rule2 = ConformalSizeAbstainer(max_set_size=1)
        combined = CombinedAbstainer(rules=[rule1, rule2], mode="all")

        proba = np.array([[0.7, 0.3], [0.55, 0.45], [0.9, 0.1]])
        conformal_sets = [[0, 1], [0, 1], [0]]

        mask, reasons = combined.apply(proba, conformal_sets)

        # Sample 0: prob OK (False), set large (True) -> False
        # Sample 1: prob low (True), set large (True) -> True
        # Sample 2: prob OK (False), set small (False) -> False
        assert mask[0] == False
        assert mask[1] == True
        assert mask[2] == False

    def test_apply_reason_codes_combined(self):
        """Test combined reason codes."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        rule2 = ConformalSizeAbstainer(max_set_size=1)
        combined = CombinedAbstainer(rules=[rule1, rule2], mode="any")

        proba = np.array([[0.7, 0.3], [0.55, 0.45]])
        conformal_sets = [[0], [0, 1]]

        mask, reasons = combined.apply(proba, conformal_sets)

        assert reasons[0] == "confident|small_set"
        assert reasons[1] == "low_confidence|large_set"

    def test_apply_three_rules(self):
        """Test combining three rules."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        rule2 = ConformalSizeAbstainer(max_set_size=1)
        rule3 = MaxProbAbstainer(threshold=0.8)  # More strict
        combined = CombinedAbstainer(rules=[rule1, rule2, rule3], mode="any")

        proba = np.array([[0.7, 0.3]])
        conformal_sets = [[0]]

        mask, reasons = combined.apply(proba, conformal_sets)

        # 0.7 > 0.6 (rule1 OK), size 1 <= 1 (rule2 OK), 0.7 <= 0.8 (rule3 abstain)
        # With 'any', result is True (abstain)
        assert mask[0] == True

    def test_apply_mode_all_strict(self):
        """Test mode='all' with strict requirements."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        rule2 = ConformalSizeAbstainer(max_set_size=1)
        combined = CombinedAbstainer(rules=[rule1, rule2], mode="all")

        proba = np.array([[0.7, 0.3], [0.55, 0.45]])
        conformal_sets = [[0], [0, 1]]

        mask, reasons = combined.apply(proba, conformal_sets)

        # Sample 0: rule1 OK, rule2 OK -> False (don't abstain)
        # Sample 1: rule1 abstain, rule2 abstain -> True (both triggered)
        assert mask[0] == False
        assert mask[1] == True

    def test_evaluate_basic(self):
        """Test evaluate metrics."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        combined = CombinedAbstainer(rules=[rule1], mode="any")

        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        mask_abstain = np.array([False, True, False])

        metrics = combined.evaluate(y_true, y_pred, mask_abstain)

        assert metrics["abstain_rate"] == 1 / 3
        assert metrics["accuracy_on_answered"] == 1.0

    def test_evaluate_all_abstain(self):
        """Test evaluate when all samples abstain."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        combined = CombinedAbstainer(rules=[rule1], mode="any")

        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        mask_abstain = np.array([True, True])

        metrics = combined.evaluate(y_true, y_pred, mask_abstain)

        assert metrics["abstain_rate"] == 1.0
        assert metrics["accuracy_on_answered"] is None


class TestAbstainerThresholdBehavior:
    """Test threshold behavior as specified in acceptance criteria."""

    def test_abstain_rate_varies_with_threshold(self):
        """Test that abstain rate changes with threshold."""
        proba = np.array([[0.9, 0.1], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7]])

        abstainer_low = MaxProbAbstainer(threshold=0.3)
        mask_low, _ = abstainer_low.apply(proba)
        rate_low = np.mean(mask_low)

        abstainer_high = MaxProbAbstainer(threshold=0.8)
        mask_high, _ = abstainer_high.apply(proba)
        rate_high = np.mean(mask_high)

        # Higher threshold should result in more abstentions
        assert rate_high > rate_low

    def test_abstain_rate_monotonic(self):
        """Test that abstain rate increases monotonically with threshold."""
        proba = np.array([[0.95, 0.05], [0.75, 0.25], [0.55, 0.45], [0.35, 0.65]])

        thresholds = [0.2, 0.4, 0.6, 0.8]
        abstain_rates = []

        for threshold in thresholds:
            abstainer = MaxProbAbstainer(threshold=threshold)
            mask, _ = abstainer.apply(proba)
            abstain_rates.append(np.mean(mask))

        # Abstain rates should be monotonically increasing
        for i in range(len(abstain_rates) - 1):
            assert abstain_rates[i] <= abstain_rates[i + 1]

    def test_abstain_rate_boundary(self):
        """Test abstain rate at boundary conditions."""
        proba = np.array([[0.5, 0.5], [0.5, 0.5]])

        abstainer_just_below = MaxProbAbstainer(threshold=0.49)
        mask_below, _ = abstainer_just_below.apply(proba)
        rate_below = np.mean(mask_below)

        abstainer_at = MaxProbAbstainer(threshold=0.5)
        mask_at, _ = abstainer_at.apply(proba)
        rate_at = np.mean(mask_at)

        abstainer_just_above = MaxProbAbstainer(threshold=0.51)
        mask_above, _ = abstainer_just_above.apply(proba)
        rate_above = np.mean(mask_above)

        # Below threshold: no abstention, At threshold: full abstention, Above: full abstention
        assert rate_below < rate_at
        assert rate_at == rate_above


class TestReasonCodeStability:
    """Test reason code stability and consistency."""

    def test_reason_codes_deterministic(self):
        """Test that reason codes are deterministic."""
        proba = np.array([[0.7, 0.3], [0.55, 0.45], [0.9, 0.1]])
        abstainer = MaxProbAbstainer(threshold=0.6)

        _, reasons1 = abstainer.apply(proba)
        _, reasons2 = abstainer.apply(proba)

        assert reasons1 == reasons2

    def test_reason_codes_match_mask(self):
        """Test that reason codes match abstention mask."""
        proba = np.array([[0.7, 0.3], [0.55, 0.45], [0.9, 0.1]])
        abstainer = MaxProbAbstainer(threshold=0.6)

        mask, reasons = abstainer.apply(proba)

        for i, (m, reason) in enumerate(zip(mask, reasons)):
            if m:
                assert reason == "low_confidence"
            else:
                assert reason == "confident"

    def test_conformal_reason_codes_deterministic(self):
        """Test conformal reason codes are deterministic."""
        conformal_sets = [[0, 1], [0, 1, 2], [1]]
        abstainer = ConformalSizeAbstainer(max_set_size=2)

        _, reasons1 = abstainer.apply(None, conformal_sets)
        _, reasons2 = abstainer.apply(None, conformal_sets)

        assert reasons1 == reasons2

    def test_combined_reason_codes_format(self):
        """Test combined reason codes have correct format."""
        rule1 = MaxProbAbstainer(threshold=0.6)
        rule2 = ConformalSizeAbstainer(max_set_size=2)
        combined = CombinedAbstainer(rules=[rule1, rule2], mode="any")

        proba = np.array([[0.7, 0.3], [0.55, 0.45]])
        conformal_sets = [[0, 1], [0, 1, 2]]

        _, reasons = combined.apply(proba, conformal_sets)

        # Should be formatted as "reason1|reason2"
        assert all("|" in r for r in reasons)


class TestIntegrationAbstention:
    """Integration tests for abstention policies."""

    def test_full_pipeline_max_prob(self):
        """Test full pipeline with MaxProbAbstainer."""
        proba = np.array([[0.8, 0.2], [0.55, 0.45], [0.9, 0.1], [0.4, 0.6]])
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])

        abstainer = MaxProbAbstainer(threshold=0.7)
        mask, reasons = abstainer.apply(proba)
        metrics = abstainer.evaluate(y_true, y_pred, mask)

        assert 0 <= metrics["abstain_rate"] <= 1
        assert metrics["accuracy_on_answered"] is not None or metrics["abstain_rate"] == 1.0
        assert len(reasons) == len(mask)

    def test_full_pipeline_combined(self):
        """Test full pipeline with CombinedAbstainer."""
        proba = np.array([[0.8, 0.2], [0.55, 0.45], [0.9, 0.1]])
        conformal_sets = [[0, 1], [0, 1, 2], [0]]
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 0, 0])

        rule1 = MaxProbAbstainer(threshold=0.7)
        rule2 = ConformalSizeAbstainer(max_set_size=1)
        combined = CombinedAbstainer(rules=[rule1, rule2], mode="any")

        mask, reasons = combined.apply(proba, conformal_sets)
        metrics = combined.evaluate(y_true, y_pred, mask)

        assert metrics["abstain_rate"] >= 0
        assert metrics["accuracy_on_answered"] is not None or mask.all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
