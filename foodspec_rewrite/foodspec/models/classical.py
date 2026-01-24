"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

Classical ML model wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression


@dataclass
class LogisticRegressionClassifier:
    """Wrapper over scikit-learn LogisticRegression with save/load helpers.

    Parameters
    ----------
    C : float, default 1.0
        Inverse of regularization strength.
    penalty : str, default "l2"
        Regularization type supported by the chosen solver.
    solver : str, default "lbfgs"
        Optimizer. Defaults to 'lbfgs' which supports L2.
    max_iter : int, default 1000
        Maximum number of iterations to converge.
    random_state : int | None, default 0
        Random seed for deterministic training where applicable.
    multi_class : {"auto", "ovr", "multinomial"}, default "auto"
        Strategy for multi-class problems.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.r_[np.random.randn(20, 5) + 1, np.random.randn(20, 5) - 1]
    >>> y = np.array([1]*20 + [0]*20)
    >>> clf = LogisticRegressionClassifier(random_state=0)
    >>> clf.fit(X, y)
    LogisticRegressionClassifier(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=0, multi_class='auto')
    >>> proba = clf.predict_proba(X)
    >>> proba.shape
    (40, 2)
    >>> path = "./_tmp_lr_model.joblib"  # doctest: +SKIP
    >>> clf.save(path)                    # doctest: +SKIP
    >>> clf2 = LogisticRegressionClassifier.load(path)  # doctest: +SKIP
    >>> np.allclose(clf2.predict_proba(X), proba)       # doctest: +SKIP
    True
    """

    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_state: Optional[int] = 0
    multi_class: str = "auto"

    _model: LogisticRegression = field(init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def _make_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            multi_class=self.multi_class,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D and align with X rows")
        self._model = self._make_model()
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X = np.asarray(X, dtype=float)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        X = np.asarray(X, dtype=float)
        proba = self._model.predict_proba(X)
        return proba

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to a Joblib file.

        Raises
        ------
        RuntimeError
        If the model is not fitted.
        """

        self._ensure_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self._model, path)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionClassifier":
        """Load a serialized classifier from a Joblib file.

        Returns a new wrapper instance with the underlying model loaded
        and marked as fitted.
        """

        model = load(path)
        inst = cls()
        inst._model = model
        inst._fitted = True
        return inst

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted; call fit() first.")


__all__ = ["LogisticRegressionClassifier"]
