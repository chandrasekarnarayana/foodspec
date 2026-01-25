"""
Comprehensive tests for interpretability and permutation importance modules.

Tests cover:
- Linear coefficient extraction (binary and multiclass)
- Coefficient sorting and top-k selection
- Permutation importance computation
- Feature ranking and importance comparisons
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from foodspec.trust.interpretability import (
    extract_linear_coefficients,
    top_k_features,
    coefficient_summary,
    to_markdown_coefficients,
    compare_coefficients,
)
from foodspec.trust.permutation import (
    permutation_importance,
    permutation_importance_with_names,
    top_k_important_features,
    compare_importances,
)


class TestExtractLinearCoefficients:
    """Test linear coefficient extraction."""
    
    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        np.random.seed(42)
        n_samples, n_features = 100, 5
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        features = [f"feature_{i}" for i in range(n_features)]
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        
        return model, features
    
    @pytest.fixture
    def multiclass_data(self):
        """Generate multiclass classification data."""
        np.random.seed(42)
        n_samples, n_features = 150, 5
        X = np.random.randn(n_samples, n_features)
        # Create 3-class target: assign labels 0, 1, 2 based on feature values
        z = X[:, 0] + 0.5 * X[:, 1]
        y = np.where(z < -0.5, 0, np.where(z < 0.5, 1, 2))
        features = [f"feature_{i}" for i in range(n_features)]
        
        model = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
        model.fit(X, y)
        
        return model, features
    
    def test_binary_extraction(self, binary_data):
        """Test coefficient extraction for binary classification."""
        model, features = binary_data
        
        coef_df = extract_linear_coefficients(model, features)
        
        # Check columns
        assert "feature" in coef_df.columns
        assert "coefficient" in coef_df.columns
        assert "abs_coefficient" in coef_df.columns
        
        # Check length
        assert len(coef_df) == len(features)
        
        # Check sorting (descending absolute value)
        assert (coef_df["abs_coefficient"].diff()[1:] <= 0).all()
        
        # Check feature names
        assert set(coef_df["feature"]) == set(features)
    
    def test_multiclass_extraction(self, multiclass_data):
        """Test coefficient extraction for multiclass."""
        model, features = multiclass_data
        
        coef_df = extract_linear_coefficients(model, features)
        
        # Check columns exist
        assert "feature" in coef_df.columns
        assert "abs_coefficient" in coef_df.columns
        assert "mean_coefficient" in coef_df.columns
        
        # Check per-class columns
        for c in range(3):
            assert f"coef_class_{c}" in coef_df.columns
        
        # Check length
        assert len(coef_df) == len(features)
        
        # Check sorting
        assert (coef_df["abs_coefficient"].diff()[1:] <= 0).all()
    
    def test_mismatched_feature_names(self, binary_data):
        """Test error on feature name mismatch."""
        model, features = binary_data
        
        wrong_features = features[:3]  # Too few
        with pytest.raises(ValueError, match="feature_names length"):
            extract_linear_coefficients(model, wrong_features)
    
    def test_no_coef_attribute(self, binary_data):
        """Test error when model has no coef_ attribute."""
        from sklearn.tree import DecisionTreeClassifier
        
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        features = [f"f{i}" for i in range(5)]
        
        with pytest.raises(AttributeError, match="coef_"):
            extract_linear_coefficients(model, features)
    
    def test_coefficient_values(self, binary_data):
        """Test coefficient values are reasonable."""
        model, features = binary_data
        
        coef_df = extract_linear_coefficients(model, features)
        
        # Coefficients should be finite
        assert np.isfinite(coef_df["coefficient"]).all()
        
        # Absolute value should match absolute of coefficient
        for idx, row in coef_df.iterrows():
            assert abs(row["coefficient"] - row["abs_coefficient"]) < 1e-10 or \
                   abs(row["coefficient"] + row["abs_coefficient"]) < 1e-10


class TestTopKFeatures:
    """Test top-k feature selection."""
    
    @pytest.fixture
    def coef_df(self):
        """Create sample coefficient DataFrame."""
        data = {
            "feature": [f"f{i}" for i in range(10)],
            "coefficient": np.random.randn(10),
            "abs_coefficient": np.random.rand(10),
        }
        df = pd.DataFrame(data)
        df = df.sort_values("abs_coefficient", ascending=False)
        return df
    
    def test_top_k_basic(self, coef_df):
        """Test top-k selection."""
        top_5 = top_k_features(coef_df, k=5)
        
        assert len(top_5) == 5
        # Should be sorted
        assert (top_5["abs_coefficient"].diff()[1:] <= 0).all()
    
    def test_top_k_exceeds_length(self, coef_df):
        """Test k > number of features."""
        top_20 = top_k_features(coef_df, k=20)
        
        # Should return all available features
        assert len(top_20) == len(coef_df)
    
    def test_top_k_zero(self, coef_df):
        """Test k=0."""
        top_0 = top_k_features(coef_df, k=0)
        
        assert len(top_0) == 0
    
    def test_negative_k(self, coef_df):
        """Test negative k raises error."""
        with pytest.raises(ValueError, match="k must be non-negative"):
            top_k_features(coef_df, k=-1)
    
    def test_missing_column(self):
        """Test error when abs_coefficient column missing."""
        df = pd.DataFrame({"feature": ["f0", "f1"]})
        
        with pytest.raises(ValueError, match="abs_coefficient"):
            top_k_features(df, k=1)


class TestCoefficientFormatting:
    """Test coefficient formatting functions."""
    
    @pytest.fixture
    def coef_df(self):
        """Create sample coefficient DataFrame."""
        np.random.seed(42)
        data = {
            "feature": [f"feature_{i}" for i in range(5)],
            "coefficient": np.random.randn(5),
            "abs_coefficient": np.abs(np.random.randn(5)),
        }
        df = pd.DataFrame(data)
        df = df.sort_values("abs_coefficient", ascending=False)
        return df
    
    def test_coefficient_summary(self, coef_df):
        """Test summary string generation."""
        summary = coefficient_summary(coef_df)
        
        assert isinstance(summary, str)
        assert "Top 10" in summary
        assert "feature_" in summary
    
    def test_markdown_export(self, coef_df):
        """Test markdown table export."""
        md = to_markdown_coefficients(coef_df, k=3)
        
        assert isinstance(md, str)
        assert "|" in md  # Markdown table syntax
        assert "feature_" in md


class TestPermutationImportance:
    """Test permutation importance computation."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate data where feature 0 is strongly predictive."""
        np.random.seed(42)
        n_samples, n_features = 200, 5
        
        X = np.random.randn(n_samples, n_features)
        # Make feature 0 strongly predictive (>0.7 implies y=1)
        y = (X[:, 0] > 0.7).astype(int)
        
        # Train model on separate training data
        X_train = np.random.randn(100, n_features)
        y_train = (X_train[:, 0] > 0.7).astype(int)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        features = [f"feature_{i}" for i in range(n_features)]
        
        return model, X, y, features
    
    def test_permutation_importance_basic(self, synthetic_data):
        """Test basic permutation importance computation."""
        model, X, y, features = synthetic_data
        
        importance = permutation_importance(
            model, X, y, metric_fn=accuracy_score, n_repeats=5, seed=42
        )
        
        # Check columns
        assert "feature" in importance.columns
        assert "importance_mean" in importance.columns
        assert "importance_std" in importance.columns
        assert "baseline_metric" in importance.columns
        
        # Check length
        assert len(importance) == len(features)
        
        # Importance should be finite
        assert np.isfinite(importance["importance_mean"]).all()
    
    def test_permutation_importance_ranking(self, synthetic_data):
        """Test that strongly predictive feature ranks highest."""
        model, X, y, features = synthetic_data
        
        importance = permutation_importance(
            model, X, y, metric_fn=accuracy_score, n_repeats=10, seed=42
        )
        
        # Feature 0 should have highest importance
        top_feature = importance.iloc[0]["feature"]
        assert top_feature == 0  # Feature 0 is most important
    
    def test_deterministic_seed(self, synthetic_data):
        """Test determinism with seed."""
        model, X, y, _ = synthetic_data
        
        imp1 = permutation_importance(
            model, X, y, accuracy_score, n_repeats=5, seed=42
        )
        imp2 = permutation_importance(
            model, X, y, accuracy_score, n_repeats=5, seed=42
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(imp1, imp2)
    
    def test_different_seeds(self, synthetic_data):
        """Test different seeds give different results."""
        model, X, y, _ = synthetic_data
        
        imp1 = permutation_importance(
            model, X, y, accuracy_score, n_repeats=5, seed=42
        )
        imp2 = permutation_importance(
            model, X, y, accuracy_score, n_repeats=5, seed=123
        )
        
        # Should be different (though order might be same)
        assert not imp1["importance_mean"].equals(imp2["importance_mean"])
    
    def test_mismatched_lengths(self):
        """Test error on X/y length mismatch."""
        model = LogisticRegression(random_state=42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 60)  # Wrong length
        
        with pytest.raises(ValueError, match="mismatched"):
            permutation_importance(model, X, y, accuracy_score)


class TestPermutationImportanceWithNames:
    """Test permutation importance with feature names."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate test data."""
        np.random.seed(42)
        n_samples, n_features = 200, 5
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0.7).astype(int)
        
        X_train = np.random.randn(100, n_features)
        y_train = (X_train[:, 0] > 0.7).astype(int)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        features = ["income", "age", "debt", "employment", "score"]
        
        return model, X, y, features
    
    def test_importance_with_names(self, synthetic_data):
        """Test importance with feature names."""
        model, X, y, features = synthetic_data
        
        importance = permutation_importance_with_names(
            model, X, y, features, accuracy_score, n_repeats=5, seed=42
        )
        
        # Feature column should contain names
        assert all(f in features for f in importance["feature"])
        
        # "income" should be top (feature 0)
        assert importance.iloc[0]["feature"] == "income"
    
    def test_feature_names_mismatch(self, synthetic_data):
        """Test error on feature names mismatch."""
        model, X, y, features = synthetic_data
        
        wrong_features = features[:3]  # Too few
        with pytest.raises(ValueError, match="feature_names length"):
            permutation_importance_with_names(
                model, X, y, wrong_features, accuracy_score
            )


class TestTopKImportantFeatures:
    """Test top-k important feature selection."""
    
    @pytest.fixture
    def importance_df(self):
        """Create sample importance DataFrame."""
        data = {
            "feature": list(range(10)),
            "importance_mean": np.random.rand(10),
            "importance_std": np.random.rand(10) * 0.1,
            "baseline_metric": [0.85] * 10,
        }
        df = pd.DataFrame(data)
        df = df.sort_values("importance_mean", ascending=False)
        return df
    
    def test_top_k_selection(self, importance_df):
        """Test top-k selection."""
        top_5 = top_k_important_features(importance_df, k=5)
        
        assert len(top_5) == 5
        # Should be sorted
        assert (top_5["importance_mean"].diff()[1:] <= 0).all()
    
    def test_top_k_zero(self, importance_df):
        """Test k=0."""
        top_0 = top_k_important_features(importance_df, k=0)
        
        assert len(top_0) == 0


class TestImportanceComparison:
    """Test importance comparison utilities."""
    
    @pytest.fixture
    def two_importance_dfs(self):
        """Create two importance DataFrames."""
        np.random.seed(42)
        data1 = {
            "feature": list(range(5)),
            "importance_mean": np.array([0.5, 0.3, 0.2, 0.1, 0.05]),
            "importance_std": np.random.rand(5) * 0.05,
            "baseline_metric": [0.85] * 5,
        }
        data2 = {
            "feature": list(range(5)),
            "importance_mean": np.array([0.4, 0.35, 0.2, 0.15, 0.1]),
            "importance_std": np.random.rand(5) * 0.05,
            "baseline_metric": [0.86] * 5,
        }
        
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        
        return df1, df2
    
    def test_compare_importances(self, two_importance_dfs):
        """Test importance comparison."""
        df1, df2 = two_importance_dfs
        
        comparison = compare_importances(
            df1, df2, model_names=("baseline", "improved")
        )
        
        # Check columns
        assert "baseline_mean" in comparison.columns
        assert "improved_mean" in comparison.columns
        assert "diff_mean" in comparison.columns
        
        # Should have all features
        assert len(comparison) >= 5


class TestCoefficientComparison:
    """Test coefficient comparison utilities."""
    
    @pytest.fixture
    def two_models(self):
        """Create two models with different coefficients."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] + 0.3 * X_train[:, 1] > 0).astype(int)
        
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        model1.fit(X_train, y_train)
        
        # Train second model with different random state
        model2 = LogisticRegression(random_state=123, max_iter=1000)
        model2.fit(X_train, y_train)
        
        features = [f"f{i}" for i in range(5)]
        
        return model1, model2, features
    
    def test_compare_coefficients(self, two_models):
        """Test coefficient comparison."""
        model1, model2, features = two_models
        
        coef1 = extract_linear_coefficients(model1, features)
        coef2 = extract_linear_coefficients(model2, features)
        
        comparison = compare_coefficients(
            coef1, coef2, model_names=("seed_42", "seed_123")
        )
        
        # Check columns exist
        assert "seed_42_coef" in comparison.columns
        assert "seed_123_coef" in comparison.columns
        assert "abs_diff" in comparison.columns
