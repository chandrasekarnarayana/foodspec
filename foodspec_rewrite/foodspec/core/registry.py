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

ComponentRegistry maps protocol strings to concrete classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Type


@dataclass
class ComponentRegistry:
    """Registry for framework components keyed by category and name.

    Categories covered: preprocess, qc, features, model, splitter, plots, reporters,
    calibrators, conformal, abstain, interpretability.

    Examples
    --------
    Registering and instantiating a component:
        registry = ComponentRegistry()
        registry.register("preprocess", "normalize", Normalize)
        step = registry.create("preprocess", "normalize", method="area")
    """

    categories: Mapping[str, MutableMapping[str, Type[Any]]] = field(
        default_factory=lambda: {
            "preprocess": {},
            "qc": {},
            "features": {},
            "model": {},
            "splitter": {},
            "plots": {},
            "reporters": {},
            "calibrators": {},
            "conformal": {},
            "abstain": {},
            "interpretability": {},
        }
    )

    def register(self, category: str, name: str, cls: Type[Any]) -> None:
        """Register a concrete class under a category and name.

        Raises
        ------
        ValueError
            If the category is unknown or the name is already registered.
        """

        if category not in self.categories:
            known = ", ".join(sorted(self.categories))
            raise ValueError(f"Unknown category '{category}'. Known categories: {known}.")

        bucket = self.categories[category]
        if name in bucket:
            raise ValueError(
                f"Component '{name}' already registered in category '{category}'."
            )

        bucket[name] = cls

    def available(self, category: str) -> list[str]:
        """List available component names for a category."""

        if category not in self.categories:
            known = ", ".join(sorted(self.categories))
            raise ValueError(f"Unknown category '{category}'. Known categories: {known}.")
        return sorted(self.categories[category])

    def create(self, category: str, name: str, **params: Any) -> Any:
        """Instantiate a component with provided parameters.

        Raises
        ------
        ValueError
            If the component is not registered in the category.
        """

        if category not in self.categories:
            known = ", ".join(sorted(self.categories))
            raise ValueError(f"Unknown category '{category}'. Known categories: {known}.")

        bucket = self.categories[category]
        if name not in bucket:
            available = ", ".join(sorted(bucket)) or "<none>"
            raise ValueError(
                f"Unknown component '{name}' for category '{category}'. "
                f"Available: {available}."
            )

        cls = bucket[name]
        return cls(**params)


__all__ = ["ComponentRegistry"]


def register_default_feature_components(registry: ComponentRegistry) -> None:
    """Register core feature-engineering components into the registry.

    Components are registered under category "features" with predictable names
    so ProtocolV2 can reference them by string identifier.
    """

    from foodspec.features import (  # Imported lazily to avoid circular deps
        BandIntegration,
        FeatureUnion,
        PCAFeatureExtractor,
        PeakAreas,
        PeakHeights,
        PeakRatios,
        PLSFeatureExtractor,
        StabilitySelector,
    )

    registrations = {
        "peak_heights": PeakHeights,
        "peak_areas": PeakAreas,
        "peak_ratios": PeakRatios,
        "band_integration": BandIntegration,
        "pca": PCAFeatureExtractor,
        "pls": PLSFeatureExtractor,
        "feature_union": FeatureUnion,
        # Alias per prompt naming
        "stability_selector": StabilitySelector,
        "stability_selection_selector": StabilitySelector,
    }

    for name, cls in registrations.items():
        if name in registry.categories["features"]:
            continue
        registry.register("features", name, cls)


__all__.append("register_default_feature_components")


def register_default_model_components(registry: ComponentRegistry) -> None:
    """Register core ML model components into the registry.

    Models are registered under category "model" with predictable names
    so ProtocolV2 can reference them by string identifier.
    
    Optional dependencies (XGBoost, LightGBM) are handled gracefully - if not
    installed, they won't be registered, but other models will still work.
    """

    from foodspec.models.classical import (
        LinearSVCClassifier,
        LogisticRegressionClassifier,
        RandomForestClassifierWrapper,
        SVCClassifier,
    )
    
    # Core models (always available)
    registrations = {
        "logistic_regression": LogisticRegressionClassifier,
        "logreg": LogisticRegressionClassifier,  # Alias
        "svm": SVCClassifier,
        "linear_svm": LinearSVCClassifier,
        "random_forest": RandomForestClassifierWrapper,
        "rf": RandomForestClassifierWrapper,  # Alias
    }
    
    # Optional: XGBoost
    try:
        from foodspec.models.boosting import XGBoostClassifierWrapper
        # Test that it can be instantiated (it checks for xgboost in __post_init__)
        try:
            _ = XGBoostClassifierWrapper()
            registrations["xgboost"] = XGBoostClassifierWrapper
            registrations["xgb"] = XGBoostClassifierWrapper  # Alias
        except ImportError:
            pass  # XGBoost library not installed
    except ImportError:
        pass  # XGBoostClassifierWrapper class not available
    
    # Optional: LightGBM
    try:
        from foodspec.models.boosting import LightGBMClassifierWrapper
        # Test that it can be instantiated (it checks for lightgbm in __post_init__)
        try:
            _ = LightGBMClassifierWrapper()
            registrations["lightgbm"] = LightGBMClassifierWrapper
            registrations["lgbm"] = LightGBMClassifierWrapper  # Alias
        except ImportError:
            pass  # LightGBM library not installed
    except ImportError:
        pass  # LightGBMClassifierWrapper class not available
    
    # PLS-DA: PLS feature extraction + Logistic Regression
    # Note: We use a factory function to create PLS-DA pipeline
    from foodspec.features.chemometrics import PLSFeatureExtractor
    
    def make_pls_da(n_components: int = 5, **kwargs):
        """Factory for PLS-DA: PLS features + LogisticRegression classifier.
        
        Parameters
        ----------
        n_components : int, default 5
            Number of PLS components.
        **kwargs : dict
            Additional arguments for LogisticRegression.
        """
        from sklearn.pipeline import Pipeline
        pls = PLSFeatureExtractor(n_components=n_components)
        clf = LogisticRegressionClassifier(**kwargs)
        
        class PLSDAClassifier:
            """PLS-DA classifier combining PLS and LogisticRegression."""
            def __init__(self, pls, clf):
                self.pls = pls
                self.clf = clf
                self._is_fitted = False
            
            def fit(self, X, y):
                self.pls.fit(X, y)
                X_pls = self.pls.transform(X)
                # Handle PLSFeatureExtractor returning (Xf, feature_names)
                if isinstance(X_pls, tuple):
                    X_pls = X_pls[0]
                self.clf.fit(X_pls, y)
                self._is_fitted = True
                return self
            
            def predict_proba(self, X):
                X_pls = self.pls.transform(X)
                # Handle PLSFeatureExtractor returning (Xf, feature_names)
                if isinstance(X_pls, tuple):
                    X_pls = X_pls[0]
                return self.clf.predict_proba(X_pls)
            
            def predict(self, X):
                X_pls = self.pls.transform(X)
                # Handle PLSFeatureExtractor returning (Xf, feature_names)
                if isinstance(X_pls, tuple):
                    X_pls = X_pls[0]
                return self.clf.predict(X_pls)
        
        return PLSDAClassifier(pls, clf)
    
    registrations["pls_da"] = make_pls_da
    registrations["plsda"] = make_pls_da  # Alias
    
    for name, cls in registrations.items():
        if name in registry.categories["model"]:
            continue
        registry.register("model", name, cls)


def register_default_splitter_components(registry: ComponentRegistry) -> None:
    """Register core CV splitter components into the registry.

    Splitters are registered under category "splitter" with predictable names
    so ProtocolV2 can reference them by string identifier.
    """

    from foodspec.validation.splits import (
        LeaveOneBatchOutSplitter,
        LeaveOneGroupOutSplitter,
        LeaveOneStageOutSplitter,
    )
    
    registrations = {
        "leave_one_group_out": LeaveOneGroupOutSplitter,
        "logo": LeaveOneGroupOutSplitter,  # Alias
        "leave_one_batch_out": LeaveOneBatchOutSplitter,
        "lobo": LeaveOneBatchOutSplitter,  # Alias
        "leave_one_stage_out": LeaveOneStageOutSplitter,
        "loso": LeaveOneStageOutSplitter,  # Alias
    }
    
    for name, cls in registrations.items():
        if name in registry.categories["splitter"]:
            continue
        registry.register("splitter", name, cls)


__all__.extend([
    "register_default_model_components",
    "register_default_splitter_components",
])


def register_default_trust_components(registry: ComponentRegistry) -> None:
    """Register trust and uncertainty components into the registry.

    Categories and components:
    - calibrators: platt, isotonic
    - conformal: mondrian
    - abstain: max_prob, conformal_size, combined
    - interpretability: coefficients, permutation_importance

    This allows ProtocolV2 trust strings to instantiate concrete implementations.
    """

    from foodspec.trust.abstain import (
        CombinedAbstainer,
        ConformalSizeAbstainer,
        MaxProbAbstainer,
    )
    from foodspec.trust.calibration import IsotonicCalibrator, PlattCalibrator
    from foodspec.trust.conformal import MondrianConformalClassifier
    from foodspec.trust.interpretability import extract_linear_coefficients
    from foodspec.trust.permutation import permutation_importance

    registrations: Dict[str, Dict[str, Callable[..., Any]]] = {
        "calibrators": {
            "platt": PlattCalibrator,
            "isotonic": IsotonicCalibrator,
        },
        "conformal": {
            "mondrian": MondrianConformalClassifier,
        },
        "abstain": {
            "max_prob": MaxProbAbstainer,
            "conformal_size": ConformalSizeAbstainer,
            "combined": CombinedAbstainer,
        },
        "interpretability": {
            "coefficients": extract_linear_coefficients,
            "permutation_importance": permutation_importance,
        },
    }

    for category, mapping in registrations.items():
        for name, cls in mapping.items():
            if name in registry.categories[category]:
                continue
            registry.register(category, name, cls)


__all__.append("register_default_trust_components")

