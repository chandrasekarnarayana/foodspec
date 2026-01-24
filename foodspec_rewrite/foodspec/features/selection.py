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

Feature selection utilities: stability selection for marker panel selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.features.marker_panel import MarkerPanel
from foodspec.validation.splits import StratifiedKFoldOrGroupKFold


class Estimator(Protocol):
    """Protocol for a scikit-learn-like estimator."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def coef_(self) -> np.ndarray: ...

    @property
    def feature_importances_(self) -> np.ndarray: ...


@dataclass
class StabilitySelector:
    """Stability selection for marker panel feature selection.

    Runs repeated subsampling of the training set, fits a sparse or
    importance-based estimator, and records feature selection frequency.

    Parameters
    ----------
    estimator_factory : callable
        Function that returns an estimator when called with hyperparameters.
        Example: ``lambda **p: LogisticRegressionClassifier(penalty="l1", C=1.0, **p)``.
    n_resamples : int, default 50
        Number of subsampling resamples for stability selection.
    subsample_fraction : float, default 0.7
        Fraction of training samples used per resample (0 < f <= 1).
    selection_threshold : float, default 0.5
        Minimum selection frequency (0-1) to include a feature.
    random_state : int, default 0
        Seed for reproducible subsampling.

    Notes
    -----
    - Feature selection MUST be fit on training folds only to avoid leakage.
    - Selected indices apply to the original feature space columns.
    - Wavenumbers can be provided to map indices to physical markers.
    """

    estimator_factory: Any
    n_resamples: int = 50
    subsample_fraction: float = 0.7
    selection_threshold: float = 0.5
    random_state: int = 0

    selected_indices_: Optional[List[int]] = None
    selection_frequencies_: Optional[np.ndarray] = None
    model_params_: Dict[str, Any] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StabilitySelector":
        """Fit stability selection on training data only.

        Returns
        -------
        self : StabilitySelector
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        if n_samples != y.shape[0]:
            raise ValueError("X and y must have same length")
        if not (0.0 < self.subsample_fraction <= 1.0):
            raise ValueError("subsample_fraction must be in (0, 1]")

        rng = np.random.default_rng(self.random_state)
        counts = np.zeros(n_features, dtype=float)
        n_sub = max(1, int(self.subsample_fraction * n_samples))

        for resample_idx in range(self.n_resamples):
            idx = rng.choice(n_samples, size=n_sub, replace=False)
            X_sub, y_sub = X[idx], y[idx]
            est = self.estimator_factory()
            est.fit(X_sub, y_sub)

            # Capture estimator parameters (first successful fit) for reporting
            if resample_idx == 0:
                if hasattr(est, "get_params"):
                    try:
                        self.model_params_ = getattr(est, "get_params")()
                    except Exception:
                        self.model_params_ = {"estimator_class": est.__class__.__name__}
                else:
                    self.model_params_ = {"estimator_class": est.__class__.__name__}

            selected = self._selected_from_estimator(est)
            if selected.size > 0:
                counts[selected] += 1.0

        freqs = counts / float(self.n_resamples)
        self.selection_frequencies_ = freqs
        self.selected_indices_ = [int(i) for i in np.where(freqs >= self.selection_threshold)[0]]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce to selected columns. Requires fitting first."""
        if not self.selected_indices_:
            raise ValueError("StabilitySelector not fitted or no features selected")
        return np.asarray(X)[:, self.selected_indices_]

    def get_marker_panel(
        self,
        x_wavenumbers: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Build marker panel payload with indices and optional wavenumbers.

        Parameters
        ----------
        x_wavenumbers : sequence of float, optional
            Spectral axis values corresponding to columns of X.

        Returns
        -------
        panel : dict
            Contains 'selected_indices', 'selected_wavenumbers' (optional),
            and 'selection_frequencies' arrays.
        """
        if self.selection_frequencies_ is None or self.selected_indices_ is None:
            raise ValueError("StabilitySelector must be fitted before building marker panel")

        # Base payload
        panel: Dict[str, Any] = {
            "selected_indices": list(self.selected_indices_),
            "selection_frequencies": self.selection_frequencies_.tolist(),
            "selection_threshold": float(self.selection_threshold),
            "model_params": self.model_params_,
        }
        # Provide a sorted copy of frequencies for acceptance checks/reporting
        panel["selection_frequencies_sorted"] = sorted(panel["selection_frequencies"], reverse=True)

        # Optional names
        if x_wavenumbers is not None:
            arr = np.asarray(x_wavenumbers, dtype=float)
            panel["selected_wavenumbers"] = [float(arr[i]) for i in self.selected_indices_]
        if feature_names is not None:
            names = list(feature_names)
            panel["selected_names"] = [names[i] for i in self.selected_indices_]
        return panel

    def _selected_from_estimator(self, est: Estimator) -> np.ndarray:
        """Infer selected features from estimator (coef_ or feature_importances_)."""
        if hasattr(est, "coef_"):
            coef = np.asarray(getattr(est, "coef_"))
            if coef.ndim == 2:
                importance = np.abs(coef).max(axis=0)
            elif coef.ndim == 1:
                importance = np.abs(coef)
            else:
                importance = np.abs(coef).ravel()
            # Non-zero indicates selected for sparse models (L1)
            return np.where(importance > 0.0)[0]
        elif hasattr(est, "feature_importances_"):
            imp = np.asarray(getattr(est, "feature_importances_"))
            # Select top-k or non-zero; here treat non-zero as selected
            return np.where(imp > 0.0)[0]
        else:
            # Fallback: select none
            return np.array([], dtype=int)


def run_stability_selection_cv(
    estimator_factory: Any,
    selector: StabilitySelector,
    X: np.ndarray,
    y: np.ndarray,
    x_wavenumbers: Sequence[float] | None,
    n_splits: int = 5,
    seed: int = 0,
    groups: Sequence[object] | None = None,
    output_dir: Path | str | None = None,
    feature_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Run stability selection inside CV and save marker_panel.json.

    Selection is fit ONLY on training folds to avoid leakage. The resulting
    marker panel aggregates selection frequencies across folds.

    Returns
    -------
    payload : dict
        Marker panel payload including selected indices/wavenumbers and
        per-feature aggregate selection frequencies.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same length")

    splitter = StratifiedKFoldOrGroupKFold(n_splits=n_splits, seed=seed)
    n_features = X.shape[1]
    agg_counts = np.zeros(n_features, dtype=float)

    for fold_id, (train_idx, _test_idx) in enumerate(splitter.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        # Fit selector on training-only
        selector.fit(X_train, y_train)
        if selector.selection_frequencies_ is None:
            continue
        agg_counts += selector.selection_frequencies_

    # Average frequency across folds
    avg_freqs = agg_counts / float(n_splits)
    selected_indices = [int(i) for i in np.where(avg_freqs >= selector.selection_threshold)[0]]

    panel: Dict[str, Any] = {
        "selected_indices": selected_indices,
        "selection_frequencies": avg_freqs.tolist(),
        "n_splits": n_splits,
        "n_resamples": selector.n_resamples,
        "subsample_fraction": selector.subsample_fraction,
        "selection_threshold": selector.selection_threshold,
        "seed": seed,
    }
    if x_wavenumbers is not None:
        arr = np.asarray(x_wavenumbers, dtype=float)
        panel["selected_wavenumbers"] = [float(arr[i]) for i in selected_indices]
    if feature_names is not None:
        names_list = list(feature_names)
        panel["selected_feature_names"] = [names_list[i] for i in selected_indices]

    # Save artifact
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = ArtifactRegistry(out_dir)
        artifacts.ensure_layout()
        # Build and save MarkerPanel artifact
        mp = MarkerPanel(
            selected_feature_names=panel.get("selected_feature_names", []),
            selected_indices=selected_indices,
            selection_frequencies=panel["selection_frequencies"],
            selected_wavenumbers=panel.get("selected_wavenumbers"),
            n_splits=n_splits,
            n_resamples=selector.n_resamples,
            subsample_fraction=selector.subsample_fraction,
            selection_threshold=selector.selection_threshold,
            seed=seed,
            peak_mappings=None,
            band_mappings=None,
            created_by="StabilitySelector",
            protocol_hash="",
            extra={},
        )
        mp.save(artifacts)

    return panel


__all__ = ["StabilitySelector", "run_stability_selection_cv"]
