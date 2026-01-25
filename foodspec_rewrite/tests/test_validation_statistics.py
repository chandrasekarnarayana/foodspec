"""
Comprehensive tests for statistical utilities (bootstrap CI, ANOVA, MANOVA).

Tests cover:
- Bootstrap confidence intervals (determinism, monotonicity, edge cases)
- ANOVA with optional scipy dependency
- MANOVA placeholder
- Integration with EvaluationResult
"""

import numpy as np
import pytest

from foodspec.validation.statistics import (
    anova_on_metric,
    bootstrap_ci,
    manova_placeholder,
)


# =============================================================================
# Bootstrap CI Tests
# =============================================================================


class TestBootstrapCIBasic:
    """Test basic bootstrap CI functionality."""

    def test_returns_three_values(self):
        """Test that bootstrap_ci returns (lower, median, upper)."""
        values = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        lower, median, upper = bootstrap_ci(values, seed=42)
        
        assert isinstance(lower, float)
        assert isinstance(median, float)
        assert isinstance(upper, float)

    def test_bounds_monotonic(self):
        """Test that lower <= median <= upper."""
        values = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        lower, median, upper = bootstrap_ci(values, seed=42)
        
        assert lower <= median, f"lower={lower} > median={median}"
        assert median <= upper, f"median={median} > upper={upper}"

    def test_deterministic_with_seed(self):
        """Test that same seed produces identical results."""
        values = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        
        result1 = bootstrap_ci(values, seed=42)
        result2 = bootstrap_ci(values, seed=42)
        
        assert result1 == result2, "Results should be identical with same seed"

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        values = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        
        result1 = bootstrap_ci(values, seed=42)
        result2 = bootstrap_ci(values, seed=99)
        
        assert result1 != result2, "Results should differ with different seeds"

    def test_median_close_to_mean(self):
        """Test that median is close to mean of input values."""
        values = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        _, median, _ = bootstrap_ci(values, n_boot=5000, seed=42)
        
        mean_val = np.mean(values)
        # Bootstrap median should be close to actual mean
        assert abs(median - mean_val) < 0.05, f"median={median}, mean={mean_val}"


class TestBootstrapCIParameters:
    """Test bootstrap CI with different parameters."""

    def test_custom_n_boot(self):
        """Test that n_boot parameter works."""
        values = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        
        result1 = bootstrap_ci(values, n_boot=100, seed=42)
        result2 = bootstrap_ci(values, n_boot=2000, seed=42)
        
        # Results should differ slightly due to different n_boot
        # (but both should be valid)
        assert len(result1) == 3
        assert len(result2) == 3

    def test_custom_alpha(self):
        """Test that alpha parameter affects CI width."""
        values = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        
        # 95% CI (alpha=0.05)
        lower_95, median_95, upper_95 = bootstrap_ci(values, alpha=0.05, seed=42)
        # 90% CI (alpha=0.10) - should be narrower
        lower_90, median_90, upper_90 = bootstrap_ci(values, alpha=0.10, seed=42)
        
        width_95 = upper_95 - lower_95
        width_90 = upper_90 - lower_90
        
        assert width_90 < width_95, "90% CI should be narrower than 95% CI"
        # Medians should be close (from same data)
        assert abs(median_95 - median_90) < 0.01

    def test_per_fold_bootstrap(self):
        """Test bootstrap on per-fold metric values."""
        fold_accuracies = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        lower, median, upper = bootstrap_ci(fold_accuracies, seed=42)
        
        # CI should encompass most fold values
        assert lower < np.min(fold_accuracies) or np.isclose(lower, np.min(fold_accuracies), atol=0.05)
        assert upper > np.max(fold_accuracies) or np.isclose(upper, np.max(fold_accuracies), atol=0.05)

    def test_per_sample_bootstrap(self):
        """Test bootstrap on per-sample predictions (binary correctness)."""
        # 80% accuracy (8 correct out of 10)
        predictions = np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1])
        lower, median, upper = bootstrap_ci(predictions, seed=42)
        
        # Median should be close to 0.8
        assert 0.6 < median < 0.9, f"Expected median near 0.8, got {median}"
        # Bounds should be reasonable
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1


class TestBootstrapCIEdgeCases:
    """Test bootstrap CI edge cases."""

    def test_single_value(self):
        """Test with single value."""
        values = np.array([0.85])
        lower, median, upper = bootstrap_ci(values, seed=42)
        
        # All should equal the single value
        assert lower == 0.85
        assert median == 0.85
        assert upper == 0.85

    def test_two_values(self):
        """Test with two values."""
        values = np.array([0.80, 0.90])
        lower, median, upper = bootstrap_ci(values, seed=42)
        
        # Should work without errors
        assert lower <= median <= upper
        # Median should be close to 0.85
        assert 0.75 < median < 0.95

    def test_many_identical_values(self):
        """Test with many identical values."""
        values = np.array([0.85] * 100)
        lower, median, upper = bootstrap_ci(values, seed=42)
        
        # All should be very close to 0.85
        assert np.isclose(lower, 0.85, atol=0.01)
        assert np.isclose(median, 0.85, atol=0.01)
        assert np.isclose(upper, 0.85, atol=0.01)

    def test_high_variance_values(self):
        """Test with high variance values."""
        values = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        lower, median, upper = bootstrap_ci(values, seed=42)
        
        # Wide CI expected
        width = upper - lower
        assert width > 0.2, f"Expected wide CI, got width={width}"
        assert lower <= median <= upper


class TestBootstrapCIErrorHandling:
    """Test bootstrap CI error handling."""

    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            bootstrap_ci(np.array([]), seed=42)

    def test_invalid_alpha_raises(self):
        """Test that invalid alpha raises ValueError."""
        values = np.array([0.85, 0.87, 0.83])
        
        with pytest.raises(ValueError, match="alpha must be in"):
            bootstrap_ci(values, alpha=0, seed=42)
        
        with pytest.raises(ValueError, match="alpha must be in"):
            bootstrap_ci(values, alpha=1, seed=42)
        
        with pytest.raises(ValueError, match="alpha must be in"):
            bootstrap_ci(values, alpha=1.5, seed=42)

    def test_invalid_n_boot_raises(self):
        """Test that invalid n_boot raises ValueError."""
        values = np.array([0.85, 0.87, 0.83])
        
        with pytest.raises(ValueError, match="n_boot must be"):
            bootstrap_ci(values, n_boot=0, seed=42)
        
        with pytest.raises(ValueError, match="n_boot must be"):
            bootstrap_ci(values, n_boot=-10, seed=42)


# =============================================================================
# ANOVA Tests
# =============================================================================


def is_scipy_available():
    """Check if scipy is installed."""
    try:
        import scipy
        return True
    except ImportError:
        return False


SCIPY_AVAILABLE = is_scipy_available()


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestANOVABasic:
    """Test basic ANOVA functionality."""

    def test_anova_returns_dict(self):
        """Test that ANOVA returns proper dict structure."""
        metrics = {
            "model_A": np.array([0.85, 0.87, 0.83, 0.88, 0.86]),
            "model_B": np.array([0.80, 0.82, 0.81, 0.79, 0.83]),
            "model_C": np.array([0.90, 0.91, 0.89, 0.92, 0.90]),
        }
        
        result = anova_on_metric(metrics, factor="model")
        
        assert isinstance(result, dict)
        assert "factor" in result
        assert "f_statistic" in result
        assert "p_value" in result
        assert "df_between" in result
        assert "df_within" in result
        assert "groups" in result
        assert "n_groups" in result
        assert "total_samples" in result

    def test_anova_factor_name(self):
        """Test that factor name is stored correctly."""
        metrics = {
            "recipe_A": np.array([0.85, 0.87, 0.83]),
            "recipe_B": np.array([0.80, 0.82, 0.81]),
        }
        
        result = anova_on_metric(metrics, factor="recipe")
        
        assert result["factor"] == "recipe"

    def test_anova_degrees_of_freedom(self):
        """Test that degrees of freedom are correct."""
        metrics = {
            "A": np.array([0.85, 0.87, 0.83]),
            "B": np.array([0.80, 0.82, 0.81]),
            "C": np.array([0.90, 0.91, 0.89]),
        }
        
        result = anova_on_metric(metrics)
        
        # 3 groups -> df_between = 3 - 1 = 2
        assert result["df_between"] == 2
        # 9 total samples - 3 groups = 6
        assert result["df_within"] == 6
        assert result["n_groups"] == 3
        assert result["total_samples"] == 9

    def test_anova_significant_difference(self):
        """Test ANOVA detects significant differences."""
        # Create clearly different groups
        metrics = {
            "low": np.array([0.50, 0.51, 0.49, 0.52, 0.50]),
            "medium": np.array([0.75, 0.76, 0.74, 0.77, 0.75]),
            "high": np.array([0.95, 0.96, 0.94, 0.97, 0.95]),
        }
        
        result = anova_on_metric(metrics)
        
        # Should have very low p-value (< 0.001)
        assert result["p_value"] < 0.001
        # F-statistic should be large
        assert result["f_statistic"] > 10

    def test_anova_no_difference(self):
        """Test ANOVA with no real differences."""
        # All groups from same distribution with larger sample size for stability
        np.random.seed(42)
        metrics = {
            "A": np.random.normal(0.8, 0.02, 30),
            "B": np.random.normal(0.8, 0.02, 30),
            "C": np.random.normal(0.8, 0.02, 30),
        }
        
        result = anova_on_metric(metrics)
        
        # Should have high p-value (> 0.01) - not too strict due to randomness
        # With this seed and sample size, p-value should be reasonably high
        assert result["p_value"] > 0.01 or result["f_statistic"] < 10


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestANOVAEdgeCases:
    """Test ANOVA edge cases."""

    def test_anova_two_groups(self):
        """Test ANOVA with two groups (like t-test)."""
        metrics = {
            "A": np.array([0.85, 0.87, 0.83, 0.88]),
            "B": np.array([0.80, 0.82, 0.81, 0.79]),
        }
        
        result = anova_on_metric(metrics)
        
        # Should work
        assert result["n_groups"] == 2
        assert result["df_between"] == 1

    def test_anova_unequal_sample_sizes(self):
        """Test ANOVA with unequal group sizes."""
        metrics = {
            "A": np.array([0.85, 0.87, 0.83]),
            "B": np.array([0.80, 0.82, 0.81, 0.79, 0.83]),
            "C": np.array([0.90, 0.91]),
        }
        
        result = anova_on_metric(metrics)
        
        assert result["n_groups"] == 3
        assert result["total_samples"] == 10
        assert result["df_between"] == 2
        assert result["df_within"] == 7


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestANOVAErrorHandling:
    """Test ANOVA error handling."""

    def test_anova_empty_dict_raises(self):
        """Test that empty dict raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            anova_on_metric({})

    def test_anova_single_group_raises(self):
        """Test that single group raises ValueError."""
        metrics = {"A": np.array([0.85, 0.87, 0.83])}
        
        with pytest.raises(ValueError, match="at least 2 groups"):
            anova_on_metric(metrics)

    def test_anova_empty_group_raises(self):
        """Test that empty group raises ValueError."""
        metrics = {
            "A": np.array([0.85, 0.87, 0.83]),
            "B": np.array([]),  # Empty
        }
        
        with pytest.raises(ValueError, match="has no samples"):
            anova_on_metric(metrics)


class TestANOVAMissingDependency:
    """Test ANOVA handling regardless of scipy availability."""

    def test_anova_raises_import_error(self):
        """Test ANOVA - error when scipy missing, success when installed."""
        metrics = {
            "A": np.array([0.85, 0.87, 0.83]),
            "B": np.array([0.80, 0.82, 0.81]),
        }
        
        if SCIPY_AVAILABLE:
            # When scipy is installed, verify ANOVA works
            result = anova_on_metric(metrics)
            assert "f_statistic" in result
            assert "p_value" in result
            assert "df_between" in result
            assert "df_within" in result
        else:
            # When scipy is not installed, verify clear error
            with pytest.raises(ImportError, match="scipy is required"):
                anova_on_metric(metrics)


# =============================================================================
# MANOVA Tests
# =============================================================================


class TestMANOVAPlaceholder:
    """Test MANOVA placeholder."""

    def test_manova_raises_not_implemented(self):
        """Test that MANOVA raises NotImplementedError."""
        metrics = {}
        
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            manova_placeholder(metrics)

    def test_manova_error_message_informative(self):
        """Test that MANOVA error message is informative."""
        metrics = {}
        
        with pytest.raises(NotImplementedError) as excinfo:
            manova_placeholder(metrics, factor="model")
        
        error_msg = str(excinfo.value)
        # Should mention multi-metric
        assert "multi-metric" in error_msg.lower() or "multivariate" in error_msg.lower()
        # Should mention statsmodels
        assert "statsmodels" in error_msg.lower()
        # Should mention anova_on_metric as alternative
        assert "anova_on_metric" in error_msg


# =============================================================================
# Integration Tests
# =============================================================================


class TestBootstrapIntegrationWithEvaluation:
    """Test bootstrap CI integration with evaluation workflow."""

    def test_bootstrap_ci_format_compatible(self):
        """Test that bootstrap_ci returns format compatible with EvaluationResult."""
        values = np.array([0.85, 0.87, 0.83, 0.88, 0.86])
        result = bootstrap_ci(values, seed=42)
        
        # Should return 3-tuple
        assert len(result) == 3
        lower, median, upper = result
        
        # All should be floats
        assert isinstance(lower, float)
        assert isinstance(median, float)
        assert isinstance(upper, float)

    def test_bootstrap_ci_with_evaluation_result_schema(self):
        """Test that bootstrap CI fits EvaluationResult schema."""
        # Simulate per-fold metrics
        fold_metrics = [
            {"fold_id": 0, "accuracy": 0.85, "macro_f1": 0.83},
            {"fold_id": 1, "accuracy": 0.87, "macro_f1": 0.85},
            {"fold_id": 2, "accuracy": 0.83, "macro_f1": 0.81},
            {"fold_id": 3, "accuracy": 0.88, "macro_f1": 0.86},
            {"fold_id": 4, "accuracy": 0.86, "macro_f1": 0.84},
        ]
        
        # Compute CIs for each metric
        bootstrap_ci_dict = {}
        for metric in ["accuracy", "macro_f1"]:
            values = np.array([fold[metric] for fold in fold_metrics])
            bootstrap_ci_dict[metric] = bootstrap_ci(values, seed=42)
        
        # Verify structure
        assert "accuracy" in bootstrap_ci_dict
        assert "macro_f1" in bootstrap_ci_dict
        assert len(bootstrap_ci_dict["accuracy"]) == 3
        assert len(bootstrap_ci_dict["macro_f1"]) == 3


class TestStatisticsModuleCompleteness:
    """Test that statistics module has all required functions."""

    def test_module_exports_bootstrap_ci(self):
        """Test that bootstrap_ci is exported."""
        from foodspec.validation.statistics import bootstrap_ci
        assert callable(bootstrap_ci)

    def test_module_exports_anova_on_metric(self):
        """Test that anova_on_metric is exported."""
        from foodspec.validation.statistics import anova_on_metric
        assert callable(anova_on_metric)

    def test_module_exports_manova_placeholder(self):
        """Test that manova_placeholder is exported."""
        from foodspec.validation.statistics import manova_placeholder
        assert callable(manova_placeholder)
