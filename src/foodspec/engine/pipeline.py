"""Pipeline orchestration for preprocessing (engine namespace)."""
from __future__ import annotations

from typing import Sequence

from foodspec.engine.preprocessing.engine import PreprocessPipeline, Step
from foodspec.data_objects.spectra_set import FoodSpectrumSet


def run_preprocessing_pipeline(ds: FoodSpectrumSet, steps: Sequence[Step]) -> FoodSpectrumSet:
    """Run a preprocessing pipeline on a dataset.

    Args:
        ds: Input dataset.
        steps: Sequence of preprocessing Step instances.

    Returns:
        Transformed dataset.
    """

    pipeline = PreprocessPipeline(steps=list(steps))
    return pipeline.fit_transform(ds)


__all__ = ["run_preprocessing_pipeline"]
