from __future__ import annotations

"""Optional deep learning models for spectral classification.

Provides a minimal 1D CNN classifier with a scikit-learn-like API. This module
relies on TensorFlow/Keras or PyTorch; it is only intended for advanced users
and examples. Dependencies are optional and must be installed separately.
"""


import importlib.util
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["Conv1DSpectrumClassifier"]


class Conv1DSpectrumClassifier(BaseEstimator, ClassifierMixin):
    """Opinionated 1D CNN classifier for spectra (optional dependency: Keras).

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    validation_split : float
        Fraction of training data used for validation.
    random_state : int | None
        Optional random seed.

    Notes
    -----
    - Uses a fixed shallow 1D CNN architecture.
    - Requires TensorFlow/Keras (`pip install tensorflow`).
    - Not used in core workflows; for advanced experimentation only.
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.1,
        random_state: Optional[int] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.random_state = random_state
        self.model_ = None
        self.classes_: Optional[np.ndarray] = None
        if importlib.util.find_spec("tensorflow") is None:
            raise ImportError(
                "Conv1DSpectrumClassifier requires TensorFlow. "
                "Please install the deep extra: pip install 'foodspec[deep]'."
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "Conv1DSpectrumClassifier requires TensorFlow. "
                "Please install the deep extra: pip install 'foodspec[deep]'."
            ) from exc

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers).")
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)

        if self.random_state is not None:
            try:
                tf.keras.utils.set_random_seed(self.random_state)
            except Exception:
                pass

        n_points = X.shape[1]
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_points, 1)),
                tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same"),
                tf.keras.layers.MaxPool1D(pool_size=2),
                tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same"),
                tf.keras.layers.MaxPool1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(len(self.classes_), activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(
            X[..., None],
            y_idx,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=0,
        )
        self.model_ = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_wavenumbers).")
        probs = self.model_.predict(X[..., None], verbose=0)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
