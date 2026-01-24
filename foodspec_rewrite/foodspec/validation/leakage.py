"""
Leakage detection utilities to prevent fitting transforms on full data before CV.

Use these assertions to ensure preprocessing or feature selection only fits on
training folds, unless explicitly marked stateless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def mark_stateless(obj: Any) -> None:
    """Mark an object as stateless (safe to fit on full data without leakage).

    Examples
    --------
    >>> class Identity:
    ...     pass
    >>> t = Identity()
    >>> mark_stateless(t)
    >>> assert getattr(t, "stateless", False)
    """
    setattr(obj, "stateless", True)


@dataclass
class FitLeakageGuard:
    """Guard to assert fitting is done on training-only data.

    Parameters
    ----------
    n_full_samples : int
        Number of samples in the full dataset (pre-split).
    stateless : bool, default False
        If True, bypass leakage checks (e.g., identity transforms).
    """

    n_full_samples: int
    stateless: bool = False

    def assert_fit(self, X_fit: np.ndarray) -> None:
        """Assert that fit() is NOT called on the full dataset unless stateless.

        Raises
        ------
        AssertionError
            If fitting is attempted on the full dataset and the transform
            is not marked stateless.
        """
        if self.stateless:
            return
        n = np.asarray(X_fit).shape[0]
        if n == self.n_full_samples:
            raise AssertionError(
                "Leakage detected: attempted to fit on full dataset before CV. "
                "Fit should be performed on training folds only or mark transform as stateless."
            )


__all__ = ["mark_stateless", "FitLeakageGuard"]
