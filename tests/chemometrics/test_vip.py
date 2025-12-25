"""Tests for Variable Importance in Projection (VIP) calculations."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from foodspec.chemometrics.vip import calculate_vip, calculate_vip_da, interpret_vip


class TestCalculateVIP:
    """Tests for calculate_vip function."""

    def test_basic_vip_calculation(self):
        """Test basic VIP calculation with simple data."""
        # Create synthetic data where first two features are important
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1

        # Fit PLS model
        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        # Calculate VIP
        vip_scores = calculate_vip(pls, X, y)

        # Assertions
        assert vip_scores.shape == (10,), "VIP scores should have shape (n_features,)"
        assert np.all(vip_scores >= 0), "VIP scores should be non-negative"

        # First two features should have higher VIP scores
        # (not strict equality due to randomness, but should be in top features)
        top_features = np.argsort(vip_scores)[-2:]
        assert 0 in top_features or 1 in top_features, "Important features should have high VIP"

    def test_vip_with_pipeline(self):
        """Test VIP calculation with sklearn Pipeline."""
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

        # Create pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression(n_components=2))])
        pipeline.fit(X, y)

        # Calculate VIP
        vip_scores = calculate_vip(pipeline, X, y)

        assert vip_scores.shape == (5,)
        assert np.all(vip_scores >= 0)

    def test_vip_single_component(self):
        """Test VIP with single component PLS."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X[:, 0] + np.random.randn(50) * 0.1

        pls = PLSRegression(n_components=1)
        pls.fit(X, y)

        vip_scores = calculate_vip(pls, X, y)

        assert vip_scores.shape == (5,)
        assert np.all(vip_scores >= 0)

    def test_vip_multitarget(self):
        """Test VIP with multi-target regression."""
        np.random.seed(42)
        X = np.random.randn(100, 8)
        y = np.column_stack([X[:, 0] + X[:, 1], X[:, 2] + X[:, 3]])

        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        vip_scores = calculate_vip(pls, X, y)

        assert vip_scores.shape == (8,)
        assert np.all(vip_scores >= 0)

    def test_vip_mathematical_property(self):
        """Test mathematical properties of VIP scores."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        vip_scores = calculate_vip(pls, X, y)

        # Mean of squared VIP scores should be approximately 1
        # (this is a theoretical property of VIP)
        mean_squared_vip = np.mean(vip_scores**2)
        assert 0.5 < mean_squared_vip < 1.5, f"Mean VIP^2 should be ~1, got {mean_squared_vip}"

    def test_vip_not_fitted_error(self):
        """Test error when model is not fitted."""
        pls = PLSRegression(n_components=2)
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        with pytest.raises(ValueError, match="must be fitted"):
            calculate_vip(pls, X, y)

    def test_vip_invalid_model_type(self):
        """Test error with invalid model type."""
        from sklearn.linear_model import LinearRegression

        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        lr = LinearRegression()
        lr.fit(X, y)

        with pytest.raises(TypeError, match="must be PLSRegression"):
            calculate_vip(lr, X, y)

    def test_vip_pipeline_without_pls(self):
        """Test error when pipeline doesn't contain PLS."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
        pipeline.fit(X, y)

        with pytest.raises(TypeError, match="Expected Pipeline's last step to be PLSRegression"):
            calculate_vip(pipeline, X, y)


class TestCalculateVIPDA:
    """Tests for calculate_vip_da function."""

    def test_basic_vip_da_binary(self):
        """Test VIP-DA for binary classification."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # For PLS-DA, fit PLS step separately (LogisticRegression expects PLS scores as input)
        pls_step = PLSRegression(n_components=3)
        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        y_encoded = lb.fit_transform(y)
        if y_encoded.shape[1] == 1:
            y_encoded = np.hstack([1 - y_encoded, y_encoded])
        pls_step.fit(X, y_encoded)

        # Create pipeline with fitted PLS
        pls_da = Pipeline([("pls", pls_step), ("clf", LogisticRegression())])
        # Fit only the classifier on PLS scores
        X_pls = pls_step.transform(X)
        pls_da.named_steps["clf"].fit(X_pls, y)

        # Calculate VIP
        vip_scores = calculate_vip_da(pls_da, X, y)

        assert vip_scores.shape == (10,)
        assert np.all(vip_scores >= 0)

    def test_vip_da_multiclass(self):
        """Test VIP-DA for multiclass classification."""
        np.random.seed(42)
        X = np.random.randn(150, 8)
        # Create 3 classes
        y = np.zeros(150, dtype=int)
        y[50:100] = 1
        y[100:] = 2

        # Fit PLS step separately
        pls_step = PLSRegression(n_components=2)
        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        y_encoded = lb.fit_transform(y)
        pls_step.fit(X, y_encoded)

        pls_da = Pipeline([("pls", pls_step), ("clf", LogisticRegression())])
        X_pls = pls_step.transform(X)
        pls_da.named_steps["clf"].fit(X_pls, y)

        vip_scores = calculate_vip_da(pls_da, X, y)

        assert vip_scores.shape == (8,)
        assert np.all(vip_scores >= 0)

    def test_vip_da_no_pls_error(self):
        """Test error when pipeline has no PLS step."""
        from sklearn.preprocessing import StandardScaler

        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipeline.fit(X, y)

        with pytest.raises(ValueError, match="must contain a PLSRegression step"):
            calculate_vip_da(pipeline, X, y)


class TestInterpretVIP:
    """Tests for interpret_vip function."""

    def test_interpret_basic(self):
        """Test basic VIP interpretation."""
        vip_scores = np.array([1.5, 0.9, 0.3, 1.8, 0.7, 1.2])
        feature_names = [f"F{i}" for i in range(6)]

        result = interpret_vip(vip_scores, feature_names)

        # Check structure
        assert "highly_important" in result
        assert "moderately_important" in result
        assert "low_importance" in result
        assert "top_10" in result
        assert "all_sorted" in result

        # Check highly important (VIP > 1.0)
        highly_important = result["highly_important"]
        assert len(highly_important) == 3  # F0, F3, F5
        assert highly_important[0][0] == "F3"  # Highest VIP
        assert highly_important[0][1] == 1.8

        # Check moderately important (0.8 < VIP <= 1.0)
        moderately_important = result["moderately_important"]
        assert len(moderately_important) == 1  # F1
        assert moderately_important[0][0] == "F1"

        # Check low importance (VIP <= 0.8)
        low_importance = result["low_importance"]
        assert len(low_importance) == 2  # F2, F4

    def test_interpret_without_names(self):
        """Test interpretation without feature names."""
        vip_scores = np.array([1.2, 0.5, 1.8])

        result = interpret_vip(vip_scores)

        # Should auto-generate names
        assert result["highly_important"][0][0] == "Feature_2"
        assert result["highly_important"][1][0] == "Feature_0"

    def test_interpret_top_10(self):
        """Test top 10 feature selection."""
        vip_scores = np.random.rand(50)
        feature_names = [f"Feature_{i}" for i in range(50)]

        result = interpret_vip(vip_scores, feature_names)

        assert len(result["top_10"]) == 10
        # Should be sorted descending
        top_10_scores = [score for _, score in result["top_10"]]
        assert top_10_scores == sorted(top_10_scores, reverse=True)

    def test_interpret_all_sorted(self):
        """Test all features are sorted."""
        vip_scores = np.array([0.5, 1.5, 0.2, 2.0, 0.8])

        result = interpret_vip(vip_scores)

        all_sorted = result["all_sorted"]
        assert len(all_sorted) == 5

        # Check descending order
        scores = [score for _, score in all_sorted]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 2.0
        assert scores[-1] == 0.2


class TestVIPIntegration:
    """Integration tests using realistic spectroscopy scenarios."""

    def test_raman_spectrum_vip(self):
        """Test VIP on simulated Raman spectrum data."""
        np.random.seed(42)

        # Simulate Raman spectra (wavenumbers from 400-1800 cm^-1)
        n_samples = 80
        n_wavenumbers = 200

        # Create synthetic spectra with peaks at specific wavenumbers
        X = np.random.randn(n_samples, n_wavenumbers) * 0.1

        # Add peaks at positions 50, 100, 150 (important wavenumbers)
        X[:, 50] += np.random.randn(n_samples) * 0.5 + 2.0
        X[:, 100] += np.random.randn(n_samples) * 0.5 + 3.0
        X[:, 150] += np.random.randn(n_samples) * 0.5 + 1.5

        # Quality score depends on these peaks
        y = 0.5 * X[:, 50] + 0.8 * X[:, 100] + 0.3 * X[:, 150] + np.random.randn(n_samples) * 0.2

        # Fit PLS model
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)

        # Calculate VIP
        vip_scores = calculate_vip(pls, X, y)

        # Important wavenumbers should have high VIP scores
        important_indices = [50, 100, 150]
        important_vips = vip_scores[important_indices]

        # At least 2 of the 3 should be > 1.0
        assert np.sum(important_vips > 1.0) >= 2, f"Important peaks should have VIP > 1, got {important_vips}"

    def test_oil_authentication_vip(self):
        """Test VIP for oil authentication (binary classification)."""
        np.random.seed(42)

        # Simulate spectra for authentic (0) and adulterated (1) oil
        n_samples = 100
        n_features = 50

        X = np.random.randn(n_samples, n_features) * 0.5

        # Adulterated oils have different profiles at features 10, 20, 30
        y = np.zeros(n_samples, dtype=int)
        y[50:] = 1  # Second half is adulterated

        X[y == 1, 10] += 2.0
        X[y == 1, 20] += 1.5
        X[y == 1, 30] -= 1.0

        # Fit PLS step separately
        pls_step = PLSRegression(n_components=3)
        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        y_encoded = lb.fit_transform(y)
        if y_encoded.shape[1] == 1:
            y_encoded = np.hstack([1 - y_encoded, y_encoded])
        pls_step.fit(X, y_encoded)

        # PLS-DA pipeline
        pls_da = Pipeline([("pls", pls_step), ("clf", LogisticRegression())])
        X_pls = pls_step.transform(X)
        pls_da.named_steps["clf"].fit(X_pls, y)

        # Calculate VIP
        vip_scores = calculate_vip_da(pls_da, X, y)

        # Discriminating features should have high VIP
        discriminating_indices = [10, 20, 30]
        discriminating_vips = vip_scores[discriminating_indices]

        assert np.mean(discriminating_vips) > np.mean(vip_scores), (
            "Discriminating features should have higher VIP than average"
        )


class TestVIPEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_vip_with_minimal_data(self):
        """Test VIP with minimal sample size."""
        np.random.seed(42)
        X = np.random.randn(10, 5)  # Only 10 samples
        y = X[:, 0] + np.random.randn(10) * 0.1

        pls = PLSRegression(n_components=2)
        pls.fit(X, y)

        vip_scores = calculate_vip(pls, X, y)

        assert vip_scores.shape == (5,)
        assert np.all(np.isfinite(vip_scores)), "VIP scores should be finite"

    def test_vip_with_many_components(self):
        """Test VIP when n_components equals n_features."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        # Use all components
        pls = PLSRegression(n_components=10)
        pls.fit(X, y)

        vip_scores = calculate_vip(pls, X, y)

        assert vip_scores.shape == (10,)
        assert np.all(np.isfinite(vip_scores))

    def test_vip_reproducibility(self):
        """Test that VIP calculation is deterministic."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)

        pls = PLSRegression(n_components=3)
        pls.fit(X, y)

        vip1 = calculate_vip(pls, X, y)
        vip2 = calculate_vip(pls, X, y)

        np.testing.assert_array_equal(vip1, vip2, err_msg="VIP calculation should be deterministic")
