from sklearn.datasets import make_classification, make_regression

from foodspec.chemometrics.models import make_pls_da, make_pls_regression
from foodspec.chemometrics.pca import run_pca
from foodspec.chemometrics.validation import (
    compute_classification_metrics,
    compute_regression_metrics,
    cross_validate_pipeline,
    permutation_test_score_wrapper,
)


def test_run_pca_shapes():
    X, _ = make_classification(n_samples=40, n_features=5, random_state=0)
    pca, result = run_pca(X, n_components=3)
    assert result.scores.shape == (40, 3)
    assert result.loadings.shape == (5, 3)
    assert len(result.explained_variance_ratio) == 3
    assert pca.n_components == 3


def test_pls_da_accuracy_on_easy_dataset():
    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=5,
        class_sep=2.5,
        flip_y=0.0,
        random_state=1,
    )
    pipeline = make_pls_da(n_components=5)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    metrics_df = compute_classification_metrics(y, preds)
    assert metrics_df["accuracy"].iloc[0] > 0.8


def test_cross_validation_wrapper_and_metrics():
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        class_sep=2.0,
        random_state=2,
    )
    pipeline = make_pls_da(n_components=3)
    cv_df = cross_validate_pipeline(pipeline, X, y, cv_splits=3, scoring="accuracy")
    assert "mean" in cv_df["fold"].values
    assert cv_df.loc[cv_df["fold"] == "mean", "score"].iloc[0] > 0.6

    # permutation test
    score, perm_scores, pvalue = permutation_test_score_wrapper(
        pipeline, X, y, scoring="accuracy", n_permutations=10, random_state=0
    )
    assert perm_scores.shape[0] == 10
    assert 0 <= pvalue <= 1


def test_regression_metrics_and_pls():
    X, y = make_regression(n_samples=80, n_features=8, n_informative=5, noise=0.1, random_state=0)
    pipeline = make_pls_regression(n_components=4)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    metrics_series = compute_regression_metrics(y, preds)
    assert metrics_series["rmse"] < 1.0
    assert metrics_series["r2"] > 0.9
