"""Model factories for chemometrics workflows."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

__all__ = [
    "make_pls_regression",
    "make_pls_da",
    "make_classifier",
    "make_mlp_regressor",
]


class _PLSProjector(BaseEstimator, TransformerMixin):
    """Project data onto PLS latent space for classification."""

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.model_: PLSRegression | None = None

    def fit(self, X, y):
        self.model_ = PLSRegression(n_components=self.n_components)
        self.model_.fit(X, y)
        return self

    def transform(self, X):
        if self.model_ is None:
            raise RuntimeError("PLSProjector not fitted.")
        out = self.model_.transform(X)
        if isinstance(out, tuple):
            x_scores = out[0]
        else:
            x_scores = out
        return x_scores

    def get_feature_names_out(self, input_features=None):
        return [f"pls_pc{i+1}" for i in range(self.n_components)]


def make_pls_regression(n_components: int = 10) -> Pipeline:
    """Create a PLS regression pipeline with scaling."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=n_components)),
        ]
    )


def make_pls_da(n_components: int = 10) -> Pipeline:
    """Create a PLS-DA (PLS + Logistic Regression) pipeline."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pls_proj", _PLSProjector(n_components=n_components)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def make_classifier(model_name: str, **kwargs: Any) -> BaseEstimator:
    """Factory for common classifiers.

    Parameters
    ----------
    model_name :
        One of: ``logreg``, ``svm_linear``, ``svm_rbf``, ``rf``, ``gboost``, ``xgb``, ``lgbm``, ``knn``, ``mlp``.
    kwargs :
        Additional parameters forwarded to the model constructor.

    Returns
    -------
    BaseEstimator
        Instantiated classifier.
    """

    name = model_name.lower()
    if name == "logreg":
        return LogisticRegression(max_iter=1000, **kwargs)
    if name == "svm_linear":
        return SVC(kernel="linear", probability=True, **kwargs)
    if name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, **kwargs)
    if name == "rf":
        return RandomForestClassifier(**kwargs)
    if name == "knn":
        return KNeighborsClassifier(**kwargs)
    if name == "gboost":
        return GradientBoostingClassifier(**kwargs)
    if name == "mlp":
        params: Dict[str, Any] = {"max_iter": 500, "hidden_layer_sizes": (100,), "random_state": 42}
        params.update(kwargs)
        return MLPClassifier(**params)
    if name == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError("xgboost is required for model_name='xgb'.") from exc
        return XGBClassifier(**kwargs)
    if name == "lgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError("lightgbm is required for model_name='lgbm'.") from exc
        return LGBMClassifier(**kwargs)

    raise ValueError(
        "model_name must be one of {'logreg','svm_linear','svm_rbf','rf','gboost','xgb','lgbm','knn','mlp'}"
    )


def make_mlp_regressor(hidden_layer_sizes: tuple[int, ...] = (100,), **kwargs: Any) -> MLPRegressor:
    """Create an MLP regressor with sensible defaults for spectral calibration.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Sizes of hidden layers.
    kwargs :
        Additional MLPRegressor kwargs.

    Returns
    -------
    MLPRegressor
        Instantiated regressor.
    """
    params: Dict[str, Any] = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "max_iter": 500,
        "random_state": 42,
    }
    params.update(kwargs)
    return MLPRegressor(**params)
