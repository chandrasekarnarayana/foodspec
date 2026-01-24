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


class UncertaintySpec(BaseModel):
    """Uncertainty and conformal settings."""

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
        """

        required_metadata = {"sample_id", "modality", "label"}
        missing_metadata = required_metadata - set(self.data.metadata_map.keys())
        if missing_metadata:
            missing = ", ".join(sorted(missing_metadata))
            raise ValueError(
                f"metadata_map is missing required keys: {missing}. "
                "Map each key to a column in your dataset."
            )

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
        """Apply sensible defaults for zero-config usage."""

        proto = self.model_copy(deep=True)

        if not proto.data.metadata_map:
            proto.data.metadata_map = {
                "sample_id": "sample_id",
                "modality": proto.data.modality,
                "label": proto.data.label,
            }

        if proto.validation.metrics == []:
            proto.validation.metrics = ["accuracy"]

        if proto.reporting.sections == []:
            proto.reporting.sections = ["summary", "metrics", "figures"]

        return proto


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
