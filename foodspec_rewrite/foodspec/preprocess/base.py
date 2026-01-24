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

Base transformer interface and a no-op IdentityTransformer.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol

import numpy as np


class Transformer(Protocol):
    """Minimal transformer interface compatible with sklearn style."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Transformer":
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        ...

    def get_params(self) -> Dict[str, Any]:
        ...

    def set_params(self, **params: Any) -> "Transformer":
        ...


class IdentityTransformer:
    """A no-op transformer useful as a placeholder.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0]])
    >>> t = IdentityTransformer()
    >>> out = t.fit_transform(X)
    >>> np.array_equal(out, X)
    True
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IdentityTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return X

    def get_params(self) -> Dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> "IdentityTransformer":
        return self


__all__ = ["Transformer", "IdentityTransformer"]
