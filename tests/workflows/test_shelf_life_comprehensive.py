"""Comprehensive tests for ShelfLifeEstimate and estimate_remaining_shelf_life."""

import numpy as np
import pandas as pd
import pytest

from foodspec.core.time import TimeSpectrumSet
from foodspec.workflows.shelf_life import ShelfLifeEstimate, estimate_remaining_shelf_life


class TestShelfLifeEstimate:
    """Test suite for ShelfLifeEstimate dataclass."""

    def test_shelf_life_estimate_creation(self):
        """Test creating a ShelfLifeEstimate instance."""
        estimate = ShelfLifeEstimate(
            entity="product_A",
            t_star=100.0,
            ci_low=95.0,
            ci_high=105.0,
            slope=0.5,
            intercept=10.0,
        )

        assert estimate.entity == "product_A"
        assert estimate.t_star == 100.0
        assert estimate.ci_low == 95.0
        assert estimate.ci_high == 105.0
        assert estimate.slope == 0.5
        assert estimate.intercept == 10.0

    def test_shelf_life_estimate_fields(self):
        """Test all ShelfLifeEstimate fields are accessible."""
        estimate = ShelfLifeEstimate(
            entity="test",
            t_star=50.0,
            ci_low=40.0,
            ci_high=60.0,
            slope=1.0,
            intercept=5.0,
        )

        # Verify all fields are present and correct type
        assert isinstance(estimate.entity, str)
        assert isinstance(estimate.t_star, float)
        assert isinstance(estimate.ci_low, float)
        assert isinstance(estimate.ci_high, float)
        assert isinstance(estimate.slope, float)
        assert isinstance(estimate.intercept, float)

    def test_shelf_life_estimate_different_values(self):
        """Test ShelfLifeEstimate with various numeric values."""
        estimates = [
            (0.0, -10.0, 10.0),
            (1000.0, 990.0, 1010.0),
            (-50.0, -60.0, -40.0),
        ]
        for t_star, ci_low, ci_high in estimates:
            estimate = ShelfLifeEstimate(
                entity=f"test_{t_star}",
                t_star=t_star,
                ci_low=ci_low,
                ci_high=ci_high,
                slope=0.1,
                intercept=1.0,
            )
            assert estimate.t_star == t_star
            assert estimate.ci_low == ci_low
            assert estimate.ci_high == ci_high


class TestEstimateRemainingSelfLife:
    """Test suite for estimate_remaining_shelf_life function."""

    def _create_time_spectrum_set(self, entities, t_points, y_func):
        """Helper to create TimeSpectrumSet."""
        X = []
        meta_rows = []
        wn = np.linspace(800, 1800, 21)
        for ent in entities:
            for t in t_points:
                X.append(np.zeros_like(wn))
                meta_rows.append(
                    {
                        "sample_id": ent,
                        "time": t,
                        "value": y_func(t, ent),
                    }
                )
        X = np.vstack(X)
        meta = pd.DataFrame(meta_rows)
        return TimeSpectrumSet(
            x=X,
            wavenumbers=wn,
            metadata=meta,
            modality="raman",
            time_col="time",
            entity_col="sample_id",
        )

    def test_single_entity(self):
        """Test with single entity."""

        def y_func(t, ent):
            return 2.0 * t + 1.0

        ds = self._create_time_spectrum_set(["X"], np.array([0, 1, 2, 3, 4], dtype=float), y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=5.0)

        assert len(result) == 1
        assert result.iloc[0]["entity"] == "X"
        assert np.isfinite(result.iloc[0]["t_star"])

    def test_multiple_entities(self):
        """Test with multiple entities."""

        def y_func(t, ent):
            offset = {"A": 0.0, "B": 0.5, "C": 1.0}.get(ent, 0.0)
            return 1.5 * t + offset

        ds = self._create_time_spectrum_set(
            ["A", "B", "C"],
            np.array([0, 1, 2, 3, 4, 5], dtype=float),
            y_func,
        )
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=4.0)

        assert len(result) == 3
        assert set(result["entity"]) == {"A", "B", "C"}

    def test_output_dataframe_structure(self):
        """Test output DataFrame has correct structure."""

        def y_func(t, ent):
            return 0.3 * t

        ds = self._create_time_spectrum_set(["P1"], np.array([0, 5, 10], dtype=float), y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=2.0)

        # Check columns
        expected_cols = {"entity", "t_star", "ci_low", "ci_high", "slope", "intercept"}
        assert set(result.columns) == expected_cols

        # Check data types
        assert result["entity"].dtype == object
        assert np.issubdtype(result["t_star"].dtype, np.number)
        assert np.issubdtype(result["ci_low"].dtype, np.number)
        assert np.issubdtype(result["ci_high"].dtype, np.number)
        assert np.issubdtype(result["slope"].dtype, np.number)
        assert np.issubdtype(result["intercept"].dtype, np.number)

    def test_different_thresholds(self):
        """Test with different threshold values."""

        def y_func(t, ent):
            return 1.0 * t

        ds = self._create_time_spectrum_set(["E"], np.array([0, 1, 2, 3, 4, 5], dtype=float), y_func)

        for threshold in [1.0, 2.5, 5.0]:
            result = estimate_remaining_shelf_life(ds, value_col="value", threshold=threshold)
            assert len(result) == 1
            # t_star should be close to threshold (since slope is 1.0, intercept is 0)
            assert np.isclose(result.iloc[0]["t_star"], threshold, atol=0.1)

    def test_ci_bounds_consistency(self):
        """Test that CI low < t_star < CI high."""

        def y_func(t, ent):
            return 0.5 * t + 0.1

        ds = self._create_time_spectrum_set(
            ["A", "B"],
            np.array([0, 1, 2, 3, 4, 5, 6], dtype=float),
            y_func,
        )
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=3.0)

        for _, row in result.iterrows():
            assert row["ci_low"] < row["t_star"] < row["ci_high"]

    def test_raises_without_entity_col(self):
        """Test that error is raised when entity_col is None."""
        wn = np.linspace(800, 1800, 21)
        X = np.zeros((5, 21))
        meta = pd.DataFrame(
            {
                "time": np.array([0, 1, 2, 3, 4], dtype=float),
                "value": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            }
        )

        ds = TimeSpectrumSet(
            x=X,
            wavenumbers=wn,
            metadata=meta,
            modality="raman",
            time_col="time",
            entity_col=None,  # No entity column
        )

        with pytest.raises(ValueError, match="entity_col must be set"):
            estimate_remaining_shelf_life(ds, value_col="value", threshold=2.0)

    def test_negative_slope(self):
        """Test with negative slope (degradation decreasing)."""

        def y_func(t, ent):
            return 10.0 - 0.5 * t  # Decreasing value

        ds = self._create_time_spectrum_set(["neg"], np.array([0, 1, 2, 3, 4], dtype=float), y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=5.0)

        assert len(result) == 1
        assert result.iloc[0]["slope"] < 0  # Negative slope
        assert np.isfinite(result.iloc[0]["t_star"])

    def test_large_dataset(self):
        """Test with larger dataset."""

        def y_func(t, ent):
            return 0.1 * t + float(ord(ent[0])) / 100.0

        entities = [f"Prod_{i}" for i in range(10)]
        t_points = np.linspace(0, 100, 50, dtype=float)
        ds = self._create_time_spectrum_set(entities, t_points, y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=5.0)

        assert len(result) == 10
        assert all(np.isfinite(result["t_star"]))
        assert all(np.isfinite(result["ci_low"]))
        assert all(np.isfinite(result["ci_high"]))

    def test_all_values_finite(self):
        """Test that all output values are finite."""

        def y_func(t, ent):
            return 0.2 * t + 0.5

        ds = self._create_time_spectrum_set(["Q1", "Q2"], np.array([0, 2, 4, 6, 8], dtype=float), y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=2.5)

        assert np.isfinite(result["t_star"]).all()
        assert np.isfinite(result["ci_low"]).all()
        assert np.isfinite(result["ci_high"]).all()
        assert np.isfinite(result["slope"]).all()
        assert np.isfinite(result["intercept"]).all()

    def test_output_values_are_float(self):
        """Test that all numeric output values are float type."""

        def y_func(t, ent):
            return 0.5 * t

        ds = self._create_time_spectrum_set(["test"], np.array([0, 1, 2, 3], dtype=float), y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=1.5)

        assert result["t_star"].dtype in [np.float32, np.float64]
        assert result["ci_low"].dtype in [np.float32, np.float64]
        assert result["ci_high"].dtype in [np.float32, np.float64]
        assert result["slope"].dtype in [np.float32, np.float64]
        assert result["intercept"].dtype in [np.float32, np.float64]

    def test_intercept_handling(self):
        """Test that intercepts are correctly calculated."""

        def y_func(t, ent):
            if ent == "X":
                return 1.0 * t + 5.0  # intercept = 5.0
            else:
                return 1.0 * t + 10.0  # intercept = 10.0

        ds = self._create_time_spectrum_set(["X", "Y"], np.array([0, 2, 4, 6], dtype=float), y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=15.0)

        # Check that intercepts are close to expected values
        result_dict = dict(zip(result["entity"], result["intercept"]))
        assert np.isclose(result_dict["X"], 5.0, atol=0.5)
        assert np.isclose(result_dict["Y"], 10.0, atol=0.5)

    def test_short_time_series(self):
        """Test with minimal time points (3 points)."""

        def y_func(t, ent):
            return 0.5 * t

        ds = self._create_time_spectrum_set(["min"], np.array([0, 5, 10], dtype=float), y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=3.0)

        assert len(result) == 1
        assert np.isfinite(result.iloc[0]["t_star"])

    def test_entity_names_preserved(self):
        """Test that entity names are preserved in output."""

        def y_func(t, ent):
            return 0.1 * t

        entities = ["Alpha", "Beta", "Gamma"]
        ds = self._create_time_spectrum_set(entities, np.array([0, 1, 2, 3], dtype=float), y_func)
        result = estimate_remaining_shelf_life(ds, value_col="value", threshold=0.2)

        assert set(result["entity"]) == set(entities)
