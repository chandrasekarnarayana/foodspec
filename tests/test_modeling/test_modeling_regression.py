import numpy as np

from foodspec.modeling import OutcomeType, build_regression_model, fit_predict
from foodspec.qc import summarize_regression_diagnostics


def test_build_regression_model_registry():
    model = build_regression_model("linear")
    assert model is not None


def test_fit_predict_regression_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.1, size=80)

    result = fit_predict(
        X,
        y,
        model_name="ridge",
        scheme="kfold",
        outer_splits=3,
        inner_splits=2,
        seed=0,
        outcome_type=OutcomeType.REGRESSION,
    )

    assert "rmse" in result.metrics
    assert result.diagnostics


def test_fit_predict_count_smoke():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = rng.poisson(lam=np.exp(0.3 * X[:, 0]))

    result = fit_predict(
        X,
        y,
        model_name="poisson",
        scheme="kfold",
        outer_splits=3,
        inner_splits=2,
        seed=0,
        outcome_type=OutcomeType.COUNT,
    )

    assert "poisson_deviance" in result.metrics
    assert "overdispersion_ratio" in result.diagnostics


def test_regression_diagnostics_helper():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.9, 2.1, 3.2, 3.8])
    diag = summarize_regression_diagnostics(y_true, y_pred, outcome_type=OutcomeType.REGRESSION)
    assert "summary" in diag
    assert not diag["residuals"].empty