"""Preprocessing recipe helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Type

from foodspec.data_objects.spectra_set import FoodSpectrumSet
from foodspec.engine.preprocessing.engine import (
    AlignmentStep,
    BaselineStep,
    DerivativeStep,
    NormalizationStep,
    PreprocessPipeline,
    ResampleStep,
    SmoothingStep,
    Step,
)

_STEP_REGISTRY: Dict[str, Type[Step]] = {
    "baseline": BaselineStep,
    "smoothing": SmoothingStep,
    "alignment": AlignmentStep,
    "normalization": NormalizationStep,
    "derivative": DerivativeStep,
    "resample": ResampleStep,
}


@dataclass
class PreprocessingRecipe:
    """Reusable preprocessing recipe built from Step instances."""

    steps: List[Step] = field(default_factory=list)
    name: str = "preprocessing_recipe"

    def add(self, step: Step) -> "PreprocessingRecipe":
        self.steps.append(step)
        return self

    def to_pipeline(self) -> PreprocessPipeline:
        return PreprocessPipeline(steps=list(self.steps))

    def run(self, ds: FoodSpectrumSet) -> FoodSpectrumSet:
        pipeline = self.to_pipeline()
        return pipeline.fit_transform(ds)

    @classmethod
    def from_config(cls, steps: Iterable[Dict[str, object]], name: Optional[str] = None) -> "PreprocessingRecipe":
        recipe = cls(name=name or "preprocessing_recipe")
        for entry in steps:
            step_name = str(entry.get("type") or entry.get("name") or "").lower()
            params = entry.get("params", {})
            if not isinstance(params, dict):
                params = {}
            step_cls = _STEP_REGISTRY.get(step_name)
            if step_cls is None:
                continue
            recipe.add(step_cls(**params))
        return recipe


__all__ = ["PreprocessingRecipe"]
