from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foodspec.chemometrics.models import make_pls_da, make_pls_regression
from foodspec.chemometrics.validation import (
    compute_classification_metrics,
    compute_regression_metrics,
    cross_validate_pipeline,
    permutation_test_score_wrapper,
)


def test_pls_models_fit_and_predict():
    Xc, yc = make_classification(
        n_samples=80, n_features=8, n_informative=5, class_sep=2.0, random_state=0
    )
    pls_da = make_pls_da(n_components=3)
    pls_da.fit(Xc, yc)
    preds = pls_da.predict(Xc)
    metrics = compute_classification_metrics(yc, preds)
    assert metrics["accuracy"].iloc[0] > 0.7

    Xr, yr = make_regression(
        n_samples=80, n_features=6, n_informative=4, noise=0.1, random_state=0
    )
    pls_reg = make_pls_regression(n_components=3)
    pls_reg.fit(Xr, yr)
    yr_pred = pls_reg.predict(Xr)
    reg_metrics = compute_regression_metrics(yr, yr_pred)
    assert reg_metrics["r2"] > 0.5


def test_cross_validate_and_permutation():
    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, class_sep=2.0, random_state=1
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    cv_df = cross_validate_pipeline(pipe, X, y, cv_splits=3, scoring="accuracy")
    fold_rows = cv_df[cv_df["fold"].apply(lambda v: isinstance(v, int))]
    assert len(fold_rows) == 3
    assert "mean" in cv_df["fold"].values and "std" in cv_df["fold"].values

    score, perm_scores, pvalue = permutation_test_score_wrapper(
        pipe, X, y, scoring="accuracy", n_permutations=20, random_state=0
    )
    assert perm_scores.shape[0] == 20
    assert 0 <= pvalue <= 1
    assert pvalue < 0.1

