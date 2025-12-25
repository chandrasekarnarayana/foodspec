"""
Protocol execution framework for FoodSpec.

Includes configuration dataclasses, step implementations, validation helpers,
and the protocol runner. Replaces the legacy protocol_engine module.
"""

from .config import ProtocolConfig, ProtocolRunResult
from .runner import ProtocolRunner
from .steps import (
    STEP_REGISTRY,
    HarmonizeStep,
    HSIRoiStep,
    HSISegmentStep,
    OutputStep,
    PreprocessStep,
    QCStep,
    RQAnalysisStep,
    Step,
)
from .utils import list_available_protocols, load_protocol, validate_protocol

__all__ = [
    "ProtocolConfig",
    "ProtocolRunResult",
    "ProtocolRunner",
    "Step",
    "PreprocessStep",
    "RQAnalysisStep",
    "OutputStep",
    "HarmonizeStep",
    "HSISegmentStep",
    "HSIRoiStep",
    "QCStep",
    "STEP_REGISTRY",
    "list_available_protocols",
    "load_protocol",
    "validate_protocol",
    "EXAMPLE_PROTOCOL",
]

# Example protocol constant preserved for backwards compatibility
EXAMPLE_PROTOCOL = {
    "name": "EdibleOil_Raman_Classification_v1",
    "description": "Raman preprocessing + RQ analysis for edible oils",
    "seed": 0,
    "steps": [
        {
            "type": "preprocess",
            "params": {
                "baseline_lambda": 1e5,
                "baseline_p": 0.01,
                "smooth_window": 9,
                "smooth_polyorder": 3,
                "normalization": "reference",
                "reference_wavenumber": 2720.0,
                "peaks": [
                    {"name": "I_1742", "wavenumber": 1742},
                    {"name": "I_1652", "wavenumber": 1652},
                    {"name": "I_1434", "wavenumber": 1434},
                    {"name": "I_1296", "wavenumber": 1296},
                    {"name": "I_1259", "wavenumber": 1259},
                    {"name": "I_2720", "wavenumber": 2720},
                ],
            },
        },
        {
            "type": "rq_analysis",
            "params": {
                "oil_col": "oil_type",
                "matrix_col": "matrix",
                "heating_col": "heating_stage",
                "random_state": 0,
                "n_splits": 5,
                "normalization_modes": ["reference", "vector", "area"],
                "ratios": [
                    {"name": "1742/2720", "numerator": "I_1742", "denominator": "I_2720"},
                    {"name": "1652/2720", "numerator": "I_1652", "denominator": "I_2720"},
                    {"name": "1434/2720", "numerator": "I_1434", "denominator": "I_2720"},
                    {"name": "1259/2720", "numerator": "I_1259", "denominator": "I_2720"},
                    {"name": "1296/2720", "numerator": "I_1296", "denominator": "I_2720"},
                ],
            },
        },
        {
            "type": "output",
            "params": {
                "output_dir": "protocol_runs",
            },
        },
    ],
}
