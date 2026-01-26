"""Model factories for chemometrics workflows."""
from __future__ import annotations


from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

from foodspec.modeling.validation.metrics import compute_vip_scores

__all__ = [
    "make_pls_regression",
    "make_pls_da",
    "make_classifier",
    "make_mlp_regressor",
    "make_simca",
    "SIMCAClassifier",
    "make_regressor",
    "make_one_class_scanner",
]


class _PLSProjector(BaseEstimator, TransformerMixin):
    """Project data onto PLS latent space for classification."""

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.model_: PLSRegression | None = None
        self.vip_scores_: np.ndarray | None = None

    def fit(self, X, y):
        self.model_ = PLSRegression(n_components=self.n_components)
        self.model_.fit(X, y)
        self.vip_scores_ = compute_vip_scores(self.model_, X, y)
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

    def get_vip_scores(self) -> np.ndarray:
        if self.model_ is None or self.vip_scores_ is None:
            raise RuntimeError("PLSProjector not fitted; call fit before requesting VIP scores.")
        return self.vip_scores_

    def get_feature_names_out(self, input_features=None):
        return [f"pls_pc{i + 1}" for i in range(self.n_components)]


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


def make_regressor(model_name: str, **kwargs: Any) -> BaseEstimator:
    """Factory for common regressors.

    Supported: ridge, lasso, elasticnet, rf_reg (RandomForestRegressor), xgb_reg, lgbm_reg, mlp_reg.
    """

    name = model_name.lower()
    if name == "ridge":
        return Ridge(**kwargs)
    if name == "lasso":
        return Lasso(**kwargs)
    if name == "elasticnet":
        return ElasticNet(**kwargs)
    if name == "rf_reg":
        return RandomForestRegressor(**kwargs)
    if name == "mlp_reg":
        params: Dict[str, Any] = {"max_iter": 500, "hidden_layer_sizes": (100,), "random_state": 42}
        params.update(kwargs)
        return MLPRegressor(**params)
    if name == "xgb_reg":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError("xgboost is required for model_name='xgb_reg'.") from exc
        return XGBRegressor(**kwargs)
    if name == "lgbm_reg":
        try:
            from lightgbm import LGBMRegressor  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError("lightgbm is required for model_name='lgbm_reg'.") from exc
        return LGBMRegressor(**kwargs)

    raise ValueError("Unsupported regressor name")


def make_one_class_scanner(model_name: str = "ocsvm", **kwargs: Any) -> BaseEstimator:
    """Factory for one-class/screening models: ocsvm, isolation_forest."""

    name = model_name.lower()
    if name in {"ocsvm", "one_class_svm"}:
        return OneClassSVM(**kwargs)
    if name in {"iforest", "isolation_forest"}:
        return IsolationForest(**kwargs)
    raise ValueError("Unsupported one-class model; use ocsvm or isolation_forest")


class SIMCAClassifier(BaseEstimator, ClassifierMixin):
    """Soft Independent Modeling of Class Analogy (SIMCA).

    Builds a PCA model per class and classifies by minimal reconstruction
    residual (Q-residual). Probabilities are inverse-residual similarities.
    """

    def __init__(self, n_components: int = 5, alpha: float = 0.975):
        self.n_components = n_components
        self.alpha = alpha
        self.models_: dict | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = {}
        for cls in self.classes_:
            Xc = X[y == cls]
            n_comp = min(self.n_components, Xc.shape[0], Xc.shape[1])
            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(Xc)
            recon = pca.inverse_transform(scores)
            residuals = np.linalg.norm(Xc - recon, axis=1)
            threshold = float(np.quantile(residuals, self.alpha))
            self.models_[cls] = {"pca": pca, "threshold": threshold}
        return self

    def _residuals(self, X):
        if self.models_ is None or self.classes_ is None:
            raise RuntimeError("SIMCAClassifier not fitted.")
        X = np.asarray(X, dtype=float)
        residuals = []
        for cls in self.classes_:
            pca = self.models_[cls]["pca"]
            scores = pca.transform(X)
            recon = pca.inverse_transform(scores)
            res = np.linalg.norm(X - recon, axis=1)
            residuals.append(res)
        return np.stack(residuals, axis=1)

    def predict(self, X):
        residuals = self._residuals(X)
        best = residuals.argmin(axis=1)
        return self.classes_[best]

    def predict_proba(self, X):
        residuals = self._residuals(X)
        inv = 1.0 / (residuals + 1e-9)
        probs = inv / inv.sum(axis=1, keepdims=True)
        return probs

    def decision_function(self, X):
        residuals = self._residuals(X)
        thresholds = np.array([self.models_[cls]["threshold"] for cls in self.classes_])
        return -(residuals - thresholds)


def make_simca(n_components: int = 5, alpha: float = 0.975) -> Pipeline:
    """Create a SIMCA classifier pipeline with scaling."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("simca", SIMCAClassifier(n_components=n_components, alpha=alpha)),
        ]
    )
