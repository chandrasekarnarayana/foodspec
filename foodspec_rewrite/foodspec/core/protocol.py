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

ProtocolV2: YAML/JSON workflow schema for FoodSpec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from foodspec.preprocess import recipes as preprocess_recipes

try:  # Optional YAML support; actionable error if missing
    import yaml
except ImportError:  # pragma: no cover - exercised in runtime, not unit tests
    yaml = None


class DataSpec(BaseModel):
    """Dataset wiring and metadata mapping."""

    model_config = ConfigDict(extra="forbid")

    input: str = Field(..., description="Path or identifier for input data")
    modality: str = Field(..., description="Measurement modality, e.g., 'raman'")
    label: str = Field(..., description="Target label column name")
    metadata_map: Dict[str, str] = Field(
        default_factory=dict, description="Mapping of canonical keys to dataset columns"
    )
    required_metadata_keys: List[str] = Field(
        default_factory=list,
        description="Metadata keys that must be present in dataset (enforced at load time)"
    )


class TaskSpec(BaseModel):
    """Task description and constraints."""

    model_config = ConfigDict(extra="forbid")

    name: str
    objective: str
    constraints: Dict[str, Any] = Field(default_factory=dict)


class PreprocessStep(BaseModel):
    """Single preprocessing step (component + params)."""

    model_config = ConfigDict(extra="forbid")

    component: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PreprocessSpec(BaseModel):
    """Preprocessing via recipe name or explicit steps."""

    model_config = ConfigDict(extra="forbid")

    recipe: Optional[str] = None
    steps: List[PreprocessStep] = Field(default_factory=list)


class QCSpec(BaseModel):
    """Quality control policy and metrics."""

    model_config = ConfigDict(extra="forbid")

    thresholds: Dict[str, float] = Field(default_factory=dict)
    metrics: List[str] = Field(default_factory=list)
    policy: str = "warn"  # warn | fail_fast
    group_by: Optional[str] = Field(
        None,
        description="Metadata key for grouping QC (e.g., 'batch', 'instrument', 'matrix')"
    )


class FeatureSpec(BaseModel):
    """Feature extraction strategy."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = "auto"
    modules: List[str] = Field(default_factory=list)


class ModelSpec(BaseModel):
    """Model family and estimator selection."""

    model_config = ConfigDict(extra="forbid")

    family: str = "sklearn"
    estimator: str = "logreg"
    params: Dict[str, Any] = Field(default_factory=dict)


class ValidationSpec(BaseModel):
    """Validation scheme and metrics."""

    model_config = ConfigDict(extra="forbid")

    scheme: str = "train_test_split"
    group_key: Optional[str] = None
    nested: bool = False
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"])
    allow_random_cv: bool = Field(
        default=False,
        description="Explicit flag to allow random CV for food data. Should be False for most use cases."
    )
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during protocol validation (e.g., random CV override)"
    )


class CalibrationSpec(BaseModel):
    """Calibration method configuration for probability calibration."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(
        default="none",
        description="Calibration method: 'none' | 'platt' | 'isotonic'",
    )

    def validate_method(self) -> None:
        """Validate calibration method is supported."""
        valid_methods = {"none", "platt", "isotonic"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid calibration method: {self.method}. "
                f"Must be one of {valid_methods}."
            )


class ConformalSpec(BaseModel):
    """Conformal prediction configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable conformal prediction for prediction set generation",
    )
    alpha: float = Field(
        default=0.1,
        description="Miscoverage rate target (e.g., 0.1 for 90% coverage)",
        ge=0.0,
        le=1.0,
    )
    condition_key: Optional[str] = Field(
        default=None,
        description="Metadata column for Mondrian stratification (e.g., 'stage', 'batch')",
    )


class AbstentionRuleSpec(BaseModel):
    """Single abstention rule configuration."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="Rule type: 'max_prob' | 'conformal_size'",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="For max_prob: confidence threshold in (0, 1]",
        gt=0.0,
        le=1.0,
    )
    max_size: Optional[int] = Field(
        default=None,
        description="For conformal_size: maximum acceptable set size",
        ge=1,
    )

    def validate_rule(self) -> None:
        """Validate rule has required parameters."""
        valid_types = {"max_prob", "conformal_size"}
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid abstention rule type: {self.type}. "
                f"Must be one of {valid_types}."
            )

        if self.type == "max_prob" and self.threshold is None:
            raise ValueError(
                "max_prob rule requires 'threshold' parameter in (0, 1]"
            )
        if self.type == "conformal_size" and self.max_size is None:
            raise ValueError(
                "conformal_size rule requires 'max_size' parameter (positive int)"
            )


class AbstentionSpec(BaseModel):
    """Abstention / selective classification configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable selective classification with abstention",
    )
    rules: List[AbstentionRuleSpec] = Field(
        default_factory=list,
        description="List of abstention rules to apply",
    )
    mode: str = Field(
        default="any",
        description="Combination mode: 'any' (OR) | 'all' (AND)",
    )

    def validate_mode(self) -> None:
        """Validate combination mode."""
        valid_modes = {"any", "all"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid abstention mode: {self.mode}. "
                f"Must be one of {valid_modes}."
            )


class TrustInterpretabilitySpec(BaseModel):
    """Interpretability and explainability methods."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable interpretability analysis",
    )
    methods: List[str] = Field(
        default_factory=list,
        description="Interpretability methods: 'coefficients' | 'permutation_importance' | 'marker_panels'",
    )

    def validate_methods(self) -> None:
        """Validate interpretability methods are supported."""
        valid_methods = {"coefficients", "permutation_importance", "marker_panels"}
        for method in self.methods:
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid interpretability method: {method}. "
                    f"Must be one of {valid_methods}."
                )


class TrustSpec(BaseModel):
    """Trust, uncertainty, and calibration configuration."""

    model_config = ConfigDict(extra="forbid")

    calibration: CalibrationSpec = Field(
        default_factory=CalibrationSpec,
        description="Probability calibration settings",
    )
    conformal: ConformalSpec = Field(
        default_factory=ConformalSpec,
        description="Conformal prediction settings",
    )
    abstention: AbstentionSpec = Field(
        default_factory=AbstentionSpec,
        description="Selective classification abstention rules",
    )
    interpretability: TrustInterpretabilitySpec = Field(
        default_factory=TrustInterpretabilitySpec,
        description="Interpretability and explainability methods",
    )

    def validate(self) -> None:
        """Validate all trust configuration."""
        self.calibration.validate_method()
        self.conformal.alpha  # Already validated by Pydantic (ge/le)

        for rule in self.abstention.rules:
            rule.validate_rule()
        self.abstention.validate_mode()

        self.interpretability.validate_methods()


class UncertaintySpec(BaseModel):
    """Uncertainty and conformal settings (legacy, for backward compatibility)."""

    model_config = ConfigDict(extra="forbid")

    conformal: Dict[str, Any] = Field(default_factory=dict)


class InterpretabilitySpec(BaseModel):
    """Interpretability methods including marker panels."""

    model_config = ConfigDict(extra="forbid")

    methods: List[str] = Field(default_factory=list)
    marker_panel: List[str] = Field(default_factory=list)


class VisualizationSpec(BaseModel):
    """Visualization expectations (plots and configs)."""

    model_config = ConfigDict(extra="forbid")

    plots: List[Dict[str, Any]] = Field(default_factory=list)


class ReportingSpec(BaseModel):
    """Reporting format and sections."""

    model_config = ConfigDict(extra="forbid")

    format: str = "markdown"
    sections: List[str] = Field(default_factory=lambda: ["summary", "metrics", "figures"])


class ExportSpec(BaseModel):
    """Export/bundle expectations."""

    model_config = ConfigDict(extra="forbid")

    bundle: Dict[str, Any] = Field(default_factory=dict)


class ComputeSpec(BaseModel):
    """Compute and cache policy."""

    model_config = ConfigDict(extra="forbid")

    cache: bool = True
    parallel: int = 1
    backend: str = "loky"


class ProtocolV2(BaseModel):
    """ProtocolV2 schema for FoodSpec workflows.

    Examples
    --------
    Loading and validating a workflow:
        protocol = ProtocolV2.load("workflow.json")
        protocol.validate(component_registry={"preprocess": {"normalize", "smooth"}})

    Expanding a recipe into explicit steps:
        expanded = protocol.expand_recipes(
            {"basic": [{"component": "normalize", "params": {"method": "area"}}]}
        )
        expanded.validate(component_registry={"preprocess": {"normalize"}})

    Persisting after defaults are applied:
        protocol.apply_defaults().dump("workflow-out.json")
    """

    model_config = ConfigDict(extra="forbid")

    version: str = "2.0.0"
    data: DataSpec
    task: TaskSpec
    preprocess: PreprocessSpec = Field(default_factory=PreprocessSpec)
    qc: QCSpec = Field(default_factory=QCSpec)
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    model: ModelSpec = Field(default_factory=ModelSpec)
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    uncertainty: UncertaintySpec = Field(default_factory=UncertaintySpec)
    trust: TrustSpec = Field(default_factory=TrustSpec, description="Trust, uncertainty, and calibration")
    interpretability: InterpretabilitySpec = Field(default_factory=InterpretabilitySpec)
    visualization: VisualizationSpec = Field(default_factory=VisualizationSpec)
    reporting: ReportingSpec = Field(default_factory=ReportingSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)
    compute: ComputeSpec = Field(default_factory=ComputeSpec)

    @classmethod
    def load(cls, path: str | Path) -> "ProtocolV2":
        """Load a protocol from YAML or JSON and apply defaults."""

        file_path = Path(path)
        text = file_path.read_text()
        suffix = file_path.suffix.lower()

        def _parse_yaml(payload: str) -> Any:
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to load YAML workflows. Install with `pip install pyyaml`."
                )
            return yaml.safe_load(payload)

        payload: Any
        if suffix in {".yaml", ".yml"}:
            payload = _parse_yaml(text)
        else:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                if yaml is None:
                    raise ValueError(
                        "Unable to parse workflow. Provide JSON, YAML, or install PyYAML for YAML support."
                    )
                payload = _parse_yaml(text)

        return cls.model_validate(payload).apply_defaults()

    def dump(self, path: str | Path) -> None:
        """Persist the protocol to YAML or JSON."""

        file_path = Path(path)
        data = self.model_dump(mode="python")
        suffix = file_path.suffix.lower()

        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to dump YAML workflows. Install with `pip install pyyaml`."
                )
            file_path.write_text(yaml.safe_dump(data, sort_keys=False))
        else:
            file_path.write_text(json.dumps(data, indent=2))

    def validate(
        self,
        component_registry: Optional[Mapping[str, Sequence[str]]] = None,
        recipe_registry: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    ) -> None:
        """Validate the protocol with actionable errors.

        - Ensures required metadata keys are mapped.
        - Flags unknown components when a registry is provided.
        - Validates recipe presence when recipes are named.
        - Enforces "No random CV for real food data" design principle.
        """

        required_metadata = {"sample_id", "modality", "label"}
        missing_metadata = required_metadata - set(self.data.metadata_map.keys())
        if missing_metadata:
            missing = ", ".join(sorted(missing_metadata))
            raise ValueError(
                f"metadata_map is missing required keys: {missing}. "
                "Map each key to a column in your dataset."
            )

        # Validate CV scheme: "No random CV for real food data"
        self._validate_cv_scheme()

        if self.preprocess.recipe and recipe_registry is not None:
            if self.preprocess.recipe not in recipe_registry:
                raise ValueError(
                    f"Unknown preprocess recipe '{self.preprocess.recipe}'. "
                    "Register it in recipe_registry before running."
                )

        registry = {k: set(v) for k, v in (component_registry or {}).items()}
        preprocess_allowed = registry.get("preprocess")
        if preprocess_allowed:
            unknown_steps = [s.component for s in self.preprocess.steps if s.component not in preprocess_allowed]
            if unknown_steps:
                listed = ", ".join(sorted(set(unknown_steps)))
                raise ValueError(
                    f"Unknown preprocess components: {listed}. Register them or update the workflow."
                )

        feature_allowed = registry.get("features")
        if feature_allowed:
            unknown_features = [m for m in self.features.modules if m not in feature_allowed]
            if unknown_features:
                listed = ", ".join(sorted(set(unknown_features)))
                raise ValueError(
                    f"Unknown feature modules: {listed}. Register them or update the workflow."
                )

        model_allowed = registry.get("model")
        if model_allowed and self.model.estimator not in model_allowed:
            raise ValueError(
                f"Unknown model estimator '{self.model.estimator}'. Register it or update the workflow."
            )

        interpretability_required = bool(self.task.constraints.get("interpretability", False))
        allow_opaque = bool(self.task.constraints.get("allow_opaque_models", False))
        if interpretability_required and not allow_opaque:
            allowed_feature_modules = {
                "peak_heights",
                "peak_areas",
                "peak_ratios",
                "band_integration",
                "pca",
                "pls",
            }
            allowed_models = {"logreg", "calibrated_logreg"}

            disallowed_features = [m for m in self.features.modules if m not in allowed_feature_modules]
            if disallowed_features:
                listed = ", ".join(sorted(set(disallowed_features)))
                allowed_list = ", ".join(sorted(allowed_feature_modules))
                raise ValueError(
                    "Interpretability constraint enabled: only interpretable feature modules are allowed. "
                    f"Allowed: {allowed_list}. Found disallowed: {listed}. "
                    "To use opaque feature pipelines, set task.constraints.allow_opaque_models=true and accept the tradeoffs."
                )

            if self.model.estimator not in allowed_models:
                allowed_models_list = ", ".join(sorted(allowed_models))
                raise ValueError(
                    "Interpretability constraint enabled: only interpretable models are allowed. "
                    f"Allowed: {allowed_models_list}. Found: {self.model.estimator}. "
                    "To use opaque models (e.g., RF/XGB), set task.constraints.allow_opaque_models=true."
                )

        # Validate trust configuration
        self.trust.validate()

    def _validate_cv_scheme(self) -> None:
        """Validate CV scheme follows 'No random CV for real food data' principle.
        
        Enforces:
        1. If metadata has batch/stage keys, default to leave_one_group_out
        2. If task is authentication/adulteration, random CV requires explicit override
        3. Random CV with food data requires allow_random_cv=True with warning
        """
        # Check if metadata has batch/stage keys
        metadata_keys = set(self.data.metadata_map.keys())
        has_batch_keys = bool(metadata_keys & {"batch", "stage", "batch_id", "stage_id"})
        
        # Check if task is critical (authentication/adulteration)
        task_name_lower = self.task.name.lower()
        task_objective_lower = self.task.objective.lower()
        is_critical_task = any(
            keyword in task_name_lower or keyword in task_objective_lower
            for keyword in ["authentication", "adulteration", "fraud", "origin"]
        )
        
        # Check if scheme is random CV
        random_cv_schemes = {"stratified_kfold", "kfold", "k_fold", "stratified_k_fold"}
        is_random_cv = self.validation.scheme.lower() in random_cv_schemes
        
        # Enforcement logic
        if is_random_cv:
            # Critical tasks with batch data absolutely require override
            if is_critical_task and has_batch_keys:
                if not self.validation.allow_random_cv:
                    raise ValueError(
                        f"Random CV ('{self.validation.scheme}') is not allowed for "
                        f"task '{self.task.name}' with batch/stage metadata. "
                        f"This violates the 'No random CV for real food data' principle.\n\n"
                        f"Reasons:\n"
                        f"  - Task type: {self.task.objective} (critical for safety/authenticity)\n"
                        f"  - Metadata contains: {sorted(metadata_keys & {'batch', 'stage', 'batch_id', 'stage_id'})}\n"
                        f"  - Random CV can leak batch effects and overestimate performance\n\n"
                        f"Recommended actions:\n"
                        f"  1. Use 'leave_one_group_out' with group_key='batch' (strongly recommended)\n"
                        f"  2. Use 'group_kfold' if you have many batches\n\n"
                        f"Only if you understand the risks and have consulted with domain experts:\n"
                        f"  - Set validation.allow_random_cv=true to override this check\n"
                        f"  - Document why random CV is acceptable for this specific use case"
                    )
                else:
                    # Override is set - record warning
                    warning_msg = (
                        f"WARNING: Random CV override active for critical task '{self.task.name}'. "
                        f"Scheme '{self.validation.scheme}' may leak batch effects. "
                        f"Performance estimates may be optimistic. Proceed with caution."
                    )
                    if warning_msg not in self.validation.validation_warnings:
                        self.validation.validation_warnings.append(warning_msg)
            
            # Any food data with batch keys should use LOBO
            elif has_batch_keys:
                if not self.validation.allow_random_cv:
                    raise ValueError(
                        f"Random CV ('{self.validation.scheme}') is not recommended for data "
                        f"with batch/stage metadata: {sorted(metadata_keys & {'batch', 'stage', 'batch_id', 'stage_id'})}\n\n"
                        f"Random CV can cause batch effects to leak between train/test splits, "
                        f"leading to overoptimistic performance estimates.\n\n"
                        f"Recommended: Use 'leave_one_group_out' with group_key='batch' or 'group_kfold'.\n\n"
                        f"To override (not recommended): Set validation.allow_random_cv=true"
                    )
                else:
                    # Override is set - record warning
                    warning_msg = (
                        f"WARNING: Random CV override active. Scheme '{self.validation.scheme}' "
                        f"used despite batch metadata. Performance may be overestimated."
                    )
                    if warning_msg not in self.validation.validation_warnings:
                        self.validation.validation_warnings.append(warning_msg)

    def expand_recipes(self, recipe_registry: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None) -> "ProtocolV2":
        """Expand preprocess.recipe into explicit steps using a registry or built-ins.

        If no recipe_registry is provided, uses built-in recipes from
        foodspec.preprocess.recipes.resolve_recipe.
        """

        if not self.preprocess.recipe:
            return self

        recipe_name = self.preprocess.recipe

        steps_payload: Sequence[Mapping[str, Any]]
        if recipe_registry and recipe_name in recipe_registry:
            steps_payload = recipe_registry[recipe_name]
        else:
            steps_payload = preprocess_recipes.resolve_recipe(recipe_name)

        steps = [PreprocessStep.model_validate(step) for step in steps_payload]
        new_preprocess = PreprocessSpec(recipe=None, steps=steps)
        return self.model_copy(update={"preprocess": new_preprocess})

    def apply_defaults(self) -> "ProtocolV2":
        """Apply sensible defaults for zero-config usage.
        
        Automatically selects leave_one_group_out when batch/stage metadata is present.
        """

        proto = self.model_copy(deep=True)

        if not proto.data.metadata_map:
            proto.data.metadata_map = {
                "sample_id": "sample_id",
                "modality": proto.data.modality,
                "label": proto.data.label,
            }

        # Auto-select LOBO if batch metadata present and scheme not explicitly set
        metadata_keys = set(proto.data.metadata_map.keys())
        has_batch_keys = bool(metadata_keys & {"batch", "stage", "batch_id", "stage_id"})
        
        if has_batch_keys and proto.validation.scheme == "train_test_split":
            # Default scheme - auto-upgrade to LOBO
            proto.validation.scheme = "leave_one_group_out"
            # Set group_key to first available batch key
            batch_keys_found = sorted(metadata_keys & {"batch", "stage", "batch_id", "stage_id"})
            if batch_keys_found:
                proto.validation.group_key = batch_keys_found[0]

        if proto.validation.metrics == []:
            proto.validation.metrics = ["accuracy"]

        if proto.reporting.sections == []:
            proto.reporting.sections = ["summary", "metrics", "figures"]

        return proto

    def infer_required_metadata_keys(self) -> List[str]:
        """Infer required metadata keys from validation scheme and QC config.

        Returns
        -------
        List[str]
            List of metadata keys that must be present in the dataset.

        Examples
        --------
        >>> protocol = ProtocolV2(
        ...     data=DataSpec(input="data.csv", modality="raman", label="target"),
        ...     task=TaskSpec(name="test", objective="classification"),
        ...     validation=ValidationSpec(scheme="group_kfold", group_key="batch")
        ... )
        >>> keys = protocol.infer_required_metadata_keys()
        >>> "batch" in keys
        True
        """
        required = []

        # Validation scheme requirements
        if self.validation.group_key:
            required.append(self.validation.group_key)

        # QC grouping requirements
        if self.qc.group_by:
            required.append(self.qc.group_by)

        return list(set(required))  # deduplicate


__all__ = [
    "ProtocolV2",
    "DataSpec",
    "TaskSpec",
    "PreprocessSpec",
    "PreprocessStep",
    "QCSpec",
    "FeatureSpec",
    "ModelSpec",
    "ValidationSpec",
    "UncertaintySpec",
    "InterpretabilitySpec",
    "VisualizationSpec",
    "ReportingSpec",
    "ExportSpec",
    "ComputeSpec",
]
