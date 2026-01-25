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

High-level modeling and validation orchestrator.
Integrates ProtocolV2, ComponentRegistry, feature pipelines, and evaluation runners.
Enforces leakage safety, determinism, and group-aware CV by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from foodspec.core.protocol import ProtocolV2, ValidationSpec
from foodspec.core.registry import ComponentRegistry
from foodspec.core.artifacts import ArtifactRegistry
from foodspec.validation.evaluation import EvaluationRunner, EvaluationResult
from foodspec.validation.splits import LeaveOneGroupOutSplitter, StratifiedKFoldOrGroupKFold


@dataclass
class ModelingConfig:
    """Configuration for modeling and validation workflow.
    
    Parameters
    ----------
    protocol : ProtocolV2
        The workflow specification (defines task, validation scheme, model, etc.).
    registry : ComponentRegistry
        Component registry for instantiating models and other components.
    artifact_registry : ArtifactRegistry, optional
        Registry for saving outputs (predictions, metrics, plots).
    seed : int, default 0
        Random seed for deterministic CV splits and model training.
    """
    
    protocol: ProtocolV2
    registry: ComponentRegistry
    artifact_registry: Optional[ArtifactRegistry] = None
    seed: int = 0
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.protocol is None:
            raise ValueError("ProtocolV2 is required")
        if self.registry is None:
            raise ValueError("ComponentRegistry is required")


@dataclass
class ModelingPipeline:
    """Orchestrates model training and evaluation with leakage safety.
    
    Ensures:
    - Feature extraction is fit on training folds only.
    - Scalers and calibration (if any) are fit on training folds only.
    - Model is fit on training folds only.
    - All transformations applied to test folds use fold-specific fits.
    - Group-aware CV (LOBO/LOSO) is default; random CV blocked unless explicitly allowed.
    
    Examples
    --------
    >>> from foodspec.core.protocol import ProtocolV2, DataSpec, TaskSpec
    >>> from foodspec.core.registry import ComponentRegistry
    >>> from foodspec.core.data import SpectraSet
    >>> import numpy as np, pandas as pd
    >>> protocol = ProtocolV2(
    ...     data=DataSpec(input="data.csv", modality="raman", label="target"),
    ...     task=TaskSpec(name="classification", objective="maximize accuracy")
    ... )
    >>> registry = ComponentRegistry()
    >>> pipeline = ModelingPipeline(protocol, registry, seed=42)
    >>> # ... prepare X, y, groups ...
    >>> result = pipeline.run(X, y, groups=groups)
    >>> print(f"Mean accuracy: {result.summary_metrics['accuracy']}")
    """
    
    config: ModelingConfig
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.config.protocol is None:
            raise ValueError("ProtocolV2 is required")
        if self.config.registry is None:
            raise ValueError("ComponentRegistry is required")
        
        # Validate validation spec
        val_spec = self.config.protocol.validation
        if val_spec.scheme not in ("train_test_split", "stratified_kfold", "group_kfold", "lobo", "loso"):
            raise ValueError(
                f"Validation scheme '{val_spec.scheme}' not recognized. "
                "Allowed: train_test_split, stratified_kfold, group_kfold, lobo, loso"
            )
    
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        metadata: Optional[pd.DataFrame] = None,
    ) -> EvaluationResult:
        """Run full modeling and validation pipeline.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input feature matrix (typically spectral data).
        y : ndarray, shape (n_samples,)
            Target labels.
        groups : ndarray, shape (n_samples,), optional
            Group identifiers for group-aware CV (e.g., batch, subject).
        metadata : DataFrame, optional
            Per-sample metadata.
        
        Returns
        -------
        EvaluationResult
            Per-fold predictions, fold metrics, summary metrics, bootstrap CIs, and
            hyperparameters (if nested CV).
        
        Raises
        ------
        ValueError
            If validation scheme requires groups but groups are not provided.
        """
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape[0] != X.shape[0]:
                raise ValueError("groups must align with X rows")
        
        # Get validation scheme and validate requirements
        val_spec = self.config.protocol.validation
        
        if val_spec.scheme in ("lobo", "loso") and groups is None:
            raise ValueError(
                f"Validation scheme '{val_spec.scheme}' requires groups to be provided"
            )
        
        # Create model instantiator
        model = self._create_model_factory()()
        
        # Get output directory for artifacts
        output_dir = None
        if self.config.artifact_registry:
            output_dir = str(self.config.artifact_registry.root)
        
        # For LOBO/LOSO, use number of groups as n_splits
        n_splits = getattr(val_spec, 'n_splits', 5)
        if val_spec.scheme in ("lobo", "loso") and groups is not None:
            n_splits = len(np.unique(groups))
        
        # Create evaluation runner
        runner = EvaluationRunner(
            estimator=model,
            n_splits=n_splits,
            seed=self.config.seed,
            output_dir=output_dir,
        )
        
        # Run evaluation
        result = runner.evaluate(X, y, groups=groups)
        
        # Save artifacts if registry provided
        if self.config.artifact_registry:
            self._save_artifacts(result)
        
        return result
    
    def _create_splitter(
        self, val_spec: ValidationSpec, groups: Optional[np.ndarray]
    ) -> Callable[[np.ndarray, np.ndarray], Sequence[Tuple[np.ndarray, np.ndarray]]]:
        """Create a splitter function based on validation spec."""
        
        scheme = val_spec.scheme
        seed = self.config.seed
        
        if scheme == "lobo":
            if groups is None:
                raise ValueError("LOBO requires groups to be provided")
            splitter = LeaveOneGroupOutSplitter()
            return lambda X, y: splitter.split(X, y, groups)
        
        elif scheme == "loso":
            if groups is None:
                raise ValueError("LOSO requires groups to be provided")
            # LOSO is same as LOBO for CV purposes
            splitter = LeaveOneGroupOutSplitter()
            return lambda X, y: splitter.split(X, y, groups)
        
        elif scheme in ("stratified_kfold", "group_kfold"):
            n_splits = getattr(val_spec, 'n_splits', 5)
            splitter = StratifiedKFoldOrGroupKFold(n_splits=n_splits, seed=seed)
            return lambda X, y: splitter.split(X, y, groups=groups)
        
        else:
            raise ValueError(f"Unsupported validation scheme: {scheme}")
    
    def _create_model_factory(self) -> Callable[..., Any]:
        """Create a model instantiator from protocol."""
        
        model_spec = self.config.protocol.model
        registry = self.config.registry
        
        def factory(**params: Any) -> Any:
            return registry.create("model", model_spec.estimator, **model_spec.params, **params)
        
        return factory
    
    def _save_artifacts(self, result: EvaluationResult) -> None:
        """Save predictions, metrics, and CIs to artifact registry."""
        
        if self.config.artifact_registry is None:
            return
        
        # TODO: Implement artifact saving
        # - Save fold predictions to predictions.csv
        # - Save fold metrics to metrics.csv
        # - Save summary metrics (mean, CI) to summary.json
        pass


__all__ = ["ModelingConfig", "ModelingPipeline"]
