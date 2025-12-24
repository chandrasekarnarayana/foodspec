"""Tests for champion-challenger comparison."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from foodspec.model_registry import (
    ChampionChallengerComparison,
    compare_champion_challenger,
)


def test_champion_challenger_no_improvement():
    """Test comparison when challenger performs equally."""
    np.random.seed(42)
    # Both models get same samples correct
    champion_scores = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0] * 10)
    challenger_scores = champion_scores + np.random.choice([-1, 0, 1], size=len(champion_scores), p=[0.05, 0.9, 0.05])
    challenger_scores = np.clip(challenger_scores, 0, 1)

    comp = compare_champion_challenger(
        champion_scores,
        challenger_scores,
        metric_name="accuracy",
        min_improvement=0.01,
    )

    assert isinstance(comp, ChampionChallengerComparison)
    assert abs(comp.improvement) < 0.1  # Small or no improvement
    assert "REJECT" in comp.recommendation or "BORDERLINE" in comp.recommendation


def test_champion_challenger_significant_improvement():
    """Test comparison with significant improvement."""
    np.random.seed(42)
    # Challenger significantly better
    champion_scores = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0] * 10)  # ~60% accuracy
    challenger_scores = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1] * 10)  # ~90% accuracy

    comp = compare_champion_challenger(
        champion_scores,
        challenger_scores,
        metric_name="accuracy",
        alpha=0.05,
        min_improvement=0.01,
    )

    assert comp.improvement > 0.1  # Strong improvement
    assert comp.is_significant  # Should be statistically significant
    assert "PROMOTE" in comp.recommendation


def test_champion_challenger_mcnemar():
    """Test McNemar's test for binary predictions."""
    np.random.seed(42)
    n = 200
    champion_scores = np.random.binomial(1, 0.75, n)
    # Challenger improves on some samples where champion failed
    challenger_scores = champion_scores.copy()
    challenger_scores[champion_scores == 0] = np.random.binomial(1, 0.5, (champion_scores == 0).sum())

    comp = compare_champion_challenger(
        champion_scores,
        challenger_scores,
        test_type="mcnemar",
        alpha=0.05,
    )

    assert comp.statistical_test == "mcnemar"
    assert comp.test_statistic >= 0
    assert 0 <= comp.pvalue <= 1


def test_champion_challenger_wilcoxon():
    """Test Wilcoxon signed-rank test."""
    np.random.seed(42)
    champion_scores = np.random.rand(100)
    challenger_scores = champion_scores + 0.05  # Slight improvement

    comp = compare_champion_challenger(
        champion_scores,
        challenger_scores,
        test_type="wilcoxon",
        alpha=0.05,
    )

    assert comp.statistical_test == "wilcoxon"
    assert comp.improvement > 0


def test_champion_challenger_confidence_interval():
    """Test confidence interval calculation."""
    np.random.seed(42)
    champion_scores = np.array([0.7, 0.8, 0.6, 0.9, 0.7] * 20)
    challenger_scores = champion_scores + 0.05

    comp = compare_champion_challenger(
        champion_scores,
        challenger_scores,
        metric_name="auc",
    )

    ci_low, ci_high = comp.confidence_interval
    assert isinstance(ci_low, float)
    assert isinstance(ci_high, float)
    assert ci_low <= comp.improvement <= ci_high


def test_champion_challenger_borderline():
    """Test borderline case (improvement but not significant)."""
    np.random.seed(42)
    n = 50  # Small sample
    champion_scores = np.random.binomial(1, 0.70, n)
    challenger_scores = np.random.binomial(1, 0.75, n)  # Slightly better

    comp = compare_champion_challenger(
        champion_scores,
        challenger_scores,
        alpha=0.05,
        min_improvement=0.01,
    )

    # May or may not be significant due to small sample
    if not comp.is_significant:
        assert "BORDERLINE" in comp.recommendation or "REJECT" in comp.recommendation


def test_champion_challenger_real_models():
    """Test comparison with actual trained models."""
    X, y = make_classification(n_samples=300, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Champion: Logistic Regression
    champion = LogisticRegression(random_state=42)
    champion.fit(X_train, y_train)
    champion_pred = champion.predict(X_test)

    # Challenger: Random Forest
    challenger = RandomForestClassifier(n_estimators=50, random_state=42)
    challenger.fit(X_train, y_train)
    challenger_pred = challenger.predict(X_test)

    # Convert predictions to binary correctness
    champion_scores = (champion_pred == y_test).astype(int)
    challenger_scores = (challenger_pred == y_test).astype(int)

    comp = compare_champion_challenger(
        champion_scores,
        challenger_scores,
        metric_name="accuracy",
        test_type="mcnemar",
        alpha=0.05,
        min_improvement=0.01,
    )

    assert 0 <= comp.champion_metric <= 1
    assert 0 <= comp.challenger_metric <= 1
    assert comp.n_samples == len(y_test)


def test_champion_challenger_improvement_pct():
    """Test improvement percentage calculation."""
    champion_scores = np.array([0.5] * 100)
    challenger_scores = np.array([0.6] * 100)  # 20% improvement

    comp = compare_champion_challenger(champion_scores, challenger_scores)

    assert comp.improvement == pytest.approx(0.1, abs=1e-6)
    assert comp.improvement_pct == pytest.approx(20.0, abs=0.1)


def test_champion_challenger_negative_improvement():
    """Test when challenger performs worse."""
    champion_scores = np.array([0.8] * 100)
    challenger_scores = np.array([0.7] * 100)  # Worse

    comp = compare_champion_challenger(champion_scores, challenger_scores)

    assert comp.improvement < 0
    assert "REJECT" in comp.recommendation
