"""Generators for manuscript-ready methods sections."""
from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


def _format_list(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


@dataclass
class MethodsConfig:
    """Structured inputs for producing reusable methods text."""

    dataset: str
    sample_size: int
    target: str
    modality: str
    instruments: List[str] = field(default_factory=list)
    acquisition: Optional[str] = None
    preprocessing: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    validation: str = "stratified 5-fold cross-validation"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    calibration: Optional[str] = None
    drift_monitoring: Optional[str] = None
    software: Optional[str] = None
    hardware: Optional[str] = None
    random_seed: Optional[int] = None
    foodspec_version: Optional[str] = None


def methods_sections(config: MethodsConfig) -> Dict[str, str]:
    """Return section-wise text blocks for manuscripts or reports."""

    instrumentation = _format_list(config.instruments) or "the configured instrumentation"
    preprocessing = _format_list(config.preprocessing) or "standard spectral preprocessing"
    models = _format_list(config.models) or "baseline classifiers"
    metrics = _format_list(config.metrics)

    sections: Dict[str, str] = {}

    sections["Samples and data collection"] = (
        f"We analyzed {config.sample_size} samples from {config.dataset}, targeting {config.target}. "
        f"Spectra were acquired using {instrumentation} under the {config.modality} modality"
        + (f" ({config.acquisition})." if config.acquisition else ".")
    )

    sections["Preprocessing"] = (
        f"Spectra were processed using {preprocessing}. "
        "All transformations were parameterized in FoodSpec configs for exact reproducibility."
    )

    sections["Modeling and calibration"] = (
        f"Models evaluated included {models}. Training followed {_format_list([config.validation])} protocols "
        f"with metrics tracked for {_format_list(config.metrics)}."
    )
    if config.calibration:
        sections["Modeling and calibration"] += f" Probability calibration used {config.calibration}."
    if config.drift_monitoring:
        sections["Modeling and calibration"] += f" Drift monitoring: {config.drift_monitoring}."

    sections["Evaluation"] = (
        f"Model performance was quantified using {config.validation} and reported as {metrics or 'standard metrics'}."
    )

    reproducibility_bits = []
    if config.software:
        reproducibility_bits.append(f"software stack: {config.software}")
    if config.foodspec_version:
        reproducibility_bits.append(f"FoodSpec version {config.foodspec_version}")
    if config.hardware:
        reproducibility_bits.append(f"hardware: {config.hardware}")
    if config.random_seed is not None:
        reproducibility_bits.append(f"random seed {config.random_seed}")
    repro_text = _format_list(reproducibility_bits) or "standard reproducibility controls"

    sections["Reproducibility"] = (
        f"Experiments were executed with {repro_text}. "
        "Pipelines, hyperparameters, and derived artifacts are exportable via foodspec.reporting utilities."
    )

    return sections


def generate_methods_text(
    config: MethodsConfig,
    style: Literal["concise", "journal", "bullet"] = "journal",
) -> str:
    """Generate narrative methods text from a MethodsConfig.

    Parameters
    ----------
    config : MethodsConfig
        Structured configuration of the study design and modeling workflow.
    style : {"concise", "journal", "bullet"}
        Output style; bullet returns a Markdown checklist-friendly format.
    """

    sections = methods_sections(config)
    if style == "bullet":
        lines = []
        for title, body in sections.items():
            lines.append(f"- **{title}:** {body}")
        return "\n".join(lines)

    paragraphs = [body for _, body in sections.items()]

    if style == "concise":
        return " ".join(paragraphs)

    return "\n\n".join(paragraphs)
