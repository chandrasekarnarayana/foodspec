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

Feature composer for chaining multiple extractors and creating hybrid feature sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from foodspec.features.base import FeatureExtractor, FeatureSet


@dataclass
class FeatureComposer:
    """Compose multiple feature extractors into a unified pipeline.

    The composer chains extractors, fits each on training data, and concatenates
    their outputs horizontally to create hybrid feature sets.

    Parameters
    ----------
    extractors : sequence of (name, extractor, kwargs)
        List of (name, extractor_instance, additional_kwargs) tuples.
        - name: string identifier for the extractor
        - extractor: instance implementing FeatureExtractor protocol
        - kwargs: dict of extra arguments passed to fit/transform (e.g., x_grid)

    Notes
    -----
    - Maintains fit/transform separation for leakage safety
    - All extractors are fit on the same training data
    - Features are concatenated in the order of extractors list

    Examples
    --------
    >>> import numpy as np
    >>> from foodspec.features.chemometrics import PCAFeatureExtractor
    >>> from foodspec.features.peaks import PeakHeights
    >>> X_train = np.random.randn(50, 100)
    >>> X_test = np.random.randn(20, 100)
    >>> x_grid = np.linspace(1000, 2000, 100)
    >>> # Compose PCA + peak features
    >>> composer = FeatureComposer([
    ...     ("pca", PCAFeatureExtractor(n_components=3), {}),
    ...     ("peaks", PeakHeights([1200, 1500]), {"x": x_grid}),
    ... ])
    >>> composer.fit(X_train, x=x_grid)
    FeatureComposer(extractors=[...])
    >>> features_train = composer.transform(X_train, x=x_grid)
    >>> features_train.features.shape[1]  # 3 PCA + 2 peaks
    5
    """

    extractors: Sequence[tuple[str, FeatureExtractor, Dict[str, Any]]]
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> "FeatureComposer":
        """Fit all extractors on training data.

        Parameters
        ----------
        X : ndarray
            Training data (n_samples, n_features).
        y : ndarray, optional
            Labels for supervised extractors (e.g., PLS).
        **kwargs : dict
            Additional parameters (e.g., x for peak-based extractors).

        Returns
        -------
        self : FeatureComposer
            Fitted composer.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        for name, extractor, extra_kwargs in self.extractors:
            # Merge extra_kwargs with fit kwargs
            fit_kwargs = {**extra_kwargs, **kwargs}
            try:
                extractor.fit(X, y, **fit_kwargs)
            except TypeError as e:
                # Some extractors may not accept y or specific kwargs
                # Try variations without offending parameters
                if "unexpected keyword argument" in str(e):
                    # Try without extra kwargs, just base params
                    try:
                        if y is not None:
                            extractor.fit(X, y)
                        else:
                            extractor.fit(X)
                    except Exception as e2:
                        raise RuntimeError(f"Failed to fit extractor '{name}': {e2}") from e2
                else:
                    raise RuntimeError(f"Failed to fit extractor '{name}': {e}") from e

        self._fitted = True
        return self

    def transform(self, X: np.ndarray, **kwargs: Any) -> FeatureSet:
        """Transform data using all fitted extractors and concatenate features.

        Parameters
        ----------
        X : ndarray
            Data to transform (n_samples, n_features).
        **kwargs : dict
            Additional parameters (e.g., x for peak-based extractors).

        Returns
        -------
        feature_set : FeatureSet
            Combined features from all extractors.

        Raises
        ------
        RuntimeError
            If transform called before fit.
        """
        if not self._fitted:
            raise RuntimeError("FeatureComposer not fitted; call fit() first")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        all_Xf: List[np.ndarray] = []
        all_feature_names: List[str] = []
        extractor_names_list: List[str] = []

        for name, extractor, extra_kwargs in self.extractors:
            transform_kwargs = {**extra_kwargs, **kwargs}
            try:
                result = extractor.transform(X, **transform_kwargs)
            except TypeError:
                # Extractor may not accept all kwargs
                try:
                    result = extractor.transform(X)
                except Exception as e:
                    raise RuntimeError(f"Failed to transform with extractor '{name}': {e}") from e

            # Handle both old DataFrame API and new tuple API
            if isinstance(result, pd.DataFrame):
                Xf = result.values
                feature_names = result.columns.tolist()
            elif isinstance(result, tuple) and len(result) == 2:
                Xf, feature_names = result
            else:
                raise TypeError(f"Extractor '{name}' must return DataFrame or (Xf, feature_names) tuple from transform()")

            all_Xf.append(Xf)
            all_feature_names.extend(feature_names)
            extractor_names_list.extend([name] * len(feature_names))

        # Concatenate all features horizontally
        combined_Xf = np.hstack(all_Xf)

        # Build metadata
        metadata: Dict[str, Any] = {
            "n_extractors": len(self.extractors),
            "extractor_names": [name for name, _, _ in self.extractors],
            "per_feature_extractors": extractor_names_list,
        }

        return FeatureSet(
            Xf=combined_Xf,
            feature_names=all_feature_names,
            feature_meta=metadata,
        )

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> FeatureSet:
        """Fit and transform in one call.

        Parameters
        ----------
        X : ndarray
            Training data.
        y : ndarray, optional
            Labels for supervised extractors.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        feature_set : FeatureSet
            Combined features.
        """
        self.fit(X, y, **kwargs)
        return self.transform(X, **kwargs)


__all__ = ["FeatureComposer"]
