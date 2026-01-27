"""
Protocol step exports.

All protocol step classes for use in protocol execution.
"""

from .base import Step
from .harmonize import HarmonizeStep
from .hsi_roi import HSIRoiStep
from .hsi_segment import HSISegmentStep
from .model_fit_predict import ModelFitPredictStep
from .multivariate import MultivariateAnalysisStep
from .output import OutputStep
from .preprocess import PreprocessStep
from .qc import QCStep
from .rq_analysis import RQAnalysisStep

__all__ = [
    "Step",
    "PreprocessStep",
    "RQAnalysisStep",
    "OutputStep",
    "HarmonizeStep",
    "HSISegmentStep",
    "HSIRoiStep",
    "QCStep",
    "ModelFitPredictStep",
    "MultivariateAnalysisStep",
]

# Step registry for protocol engine
STEP_REGISTRY = {
    PreprocessStep.name: PreprocessStep,
    RQAnalysisStep.name: RQAnalysisStep,
    OutputStep.name: OutputStep,
    HarmonizeStep.name: HarmonizeStep,
    HSISegmentStep.name: HSISegmentStep,
    HSIRoiStep.name: HSIRoiStep,
    QCStep.name: QCStep,
    ModelFitPredictStep.name: ModelFitPredictStep,
    MultivariateAnalysisStep.name: MultivariateAnalysisStep,
}
