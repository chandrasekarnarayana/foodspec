"""
Hybrid feature strategies: union of multiple feature extractors.

FeatureUnion composes multiple FeatureExtractor-compatible instances,
fitting each on training data and concatenating their transformed outputs.

Deterministic behavior depends on individual extractors' seeds/settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import FeatureExtractor, FeatureSet


def _derive_name(extractor: Any) -> str:
    """Derive a short prefix name from an extractor instance.

    Priority:
    - If extractor has attribute `union_name`, use it.
    - Else, use class name lowercased with common suffixes removed
      ("featureextractor", "extractor").
    """
    if hasattr(extractor, "union_name") and isinstance(getattr(extractor, "union_name"), str):
        return getattr(extractor, "union_name")
    cls = extractor.__class__.__name__.lower()
    for suffix in ("featureextractor", "extractor"):
        if cls.endswith(suffix):
            cls = cls[: -len(suffix)]
            break
    return cls


@dataclass
class FeatureUnion:
    """Union of multiple feature extractors with concatenated outputs.

    Parameters
    ----------
    extractors : sequence of FeatureExtractor
        List of extractor instances implementing the FeatureExtractor protocol.
    prefix : bool, default True
        If True, prefix each feature name with the extractor name
        (e.g., "pca:pca_1").

    Notes
    -----
    - Fit/transform are executed in the order provided.
    - Transform returns a FeatureSet of concatenated features.
    - Metadata tracks extractor names and per-feature origin.
    """

    extractors: Sequence[FeatureExtractor]
    prefix: bool = True
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> "FeatureUnion":
        """Fit all extractors on training data.

        Parameters
        ----------
        X : ndarray
            Training data (n_samples, n_features).
        y : ndarray, optional
            Labels for supervised extractors.
        **kwargs : dict
            Additional parameters (e.g., x for peak-based wrappers).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        for extractor in self.extractors:
            try:
                extractor.fit(X, y, **kwargs)
            except TypeError as e:
                # Fallbacks for extractors that do not accept all kwargs
                try:
                    if y is not None:
                        extractor.fit(X, y)
                    else:
                        extractor.fit(X)
                except Exception as e2:
                    raise RuntimeError(f"Failed to fit extractor '{extractor}': {e2}") from e2
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, **kwargs: Any) -> FeatureSet:
        """Transform data using all fitted extractors and concatenate features.

        Returns
        -------
        FeatureSet
            Combined features with prefixed names and merged metadata.
        """
        if not self._fitted:
            raise RuntimeError("FeatureUnion not fitted; call fit() first")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        all_Xf: List[np.ndarray] = []
        all_feature_names: List[str] = []
        per_feature_extractors: List[str] = []
        extractor_names: List[str] = []
        merged_meta: Dict[str, Any] = {}

        for extractor in self.extractors:
            name = _derive_name(extractor)
            extractor_names.append(name)

            try:
                result = extractor.transform(X, **kwargs)
            except TypeError:
                try:
                    result = extractor.transform(X)
                except Exception as e:
                    raise RuntimeError(f"Failed to transform with extractor '{name}': {e}") from e

            if isinstance(result, pd.DataFrame):
                Xf = result.values
                feature_names = result.columns.tolist()
            elif isinstance(result, tuple) and len(result) == 2:
                Xf, feature_names = result
            else:
                raise TypeError(
                    f"Extractor '{name}' must return DataFrame or (Xf, feature_names) tuple from transform()"
                )

            # Prefix names if requested
            if self.prefix:
                feature_names = [f"{name}:{fname}" for fname in feature_names]

            all_Xf.append(Xf)
            all_feature_names.extend(feature_names)
            per_feature_extractors.extend([name] * len(feature_names))

            # Merge any known per-extractor metadata (e.g., explained variance)
            # Include params for traceability
            params = getattr(extractor, "get_params", lambda: {})()
            merged_meta[f"{name}.params"] = params
            # Example: PCA explained variance ratios
            if hasattr(extractor, "explained_variance_ratio_"):
                evr = getattr(extractor, "explained_variance_ratio_")
                merged_meta[f"{name}.explained_variance_ratio_"] = list(np.asarray(evr).tolist())

        combined_Xf = np.hstack(all_Xf)
        metadata: Dict[str, Any] = {
            "n_extractors": len(self.extractors),
            "extractor_names": extractor_names,
            "per_feature_extractors": per_feature_extractors,
            "union_prefix": self.prefix,
            **merged_meta,
        }

        return FeatureSet(
            Xf=combined_Xf,
            feature_names=all_feature_names,
            feature_meta=metadata,
        )

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> FeatureSet:
        self.fit(X, y, **kwargs)
        return self.transform(X, **kwargs)


__all__ = ["FeatureUnion"]
