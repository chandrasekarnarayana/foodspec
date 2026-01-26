"""YAML recipe loading and management for preprocessing.

Supports loading YAML presets and merging with protocol and CLI overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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

# Import new Raman/FTIR operators
try:
    from foodspec.preprocess.spectroscopy_operators import (
        AtmosphericCorrectionOperator,
        DespikeOperator,
        EMSCOperator,
        FluorescenceRemovalOperator,
        InterpolationOperator,
        MSCOperator,
    )

    _EXTENDED_OPS_AVAILABLE = True
except ImportError:
    _EXTENDED_OPS_AVAILABLE = False


# Operator registry (stable names for YAML)
_OPERATOR_REGISTRY: Dict[str, type[Step]] = {
    "baseline": BaselineStep,
    "smoothing": SmoothingStep,
    "alignment": AlignmentStep,
    "normalization": NormalizationStep,
    "derivative": DerivativeStep,
    "resample": ResampleStep,
}

if _EXTENDED_OPS_AVAILABLE:
    _OPERATOR_REGISTRY.update(
        {
            "despike": DespikeOperator,
            "fluorescence_removal": FluorescenceRemovalOperator,
            "emsc": EMSCOperator,
            "msc": MSCOperator,
            "atmospheric_correction": AtmosphericCorrectionOperator,
            "interpolation": InterpolationOperator,
        }
    )


def list_operators() -> List[str]:
    """List all available operator names."""
    return list(_OPERATOR_REGISTRY.keys())


def load_preset_yaml(name: str) -> Dict[str, Any]:
    """Load preset YAML file by name.

    Parameters
    ----------
    name : str
        Preset name ('default', 'raman', 'ftir', 'oil_auth', 'chips_matrix').

    Returns
    -------
    Dict[str, Any]
        Parsed preset configuration.
    """
    presets_dir = Path(__file__).parent / "presets"

    preset_files = {
        "default": "default.yaml",
        "raman": "raman.yaml",
        "ftir": "ftir.yaml",
        "oil_auth": "custom/oil_auth.yaml",
        "chips_matrix": "custom/chips_matrix.yaml",
    }

    if name not in preset_files:
        raise ValueError(f"Unknown preset: {name}. Available: {list(preset_files.keys())}")

    preset_path = presets_dir / preset_files[name]

    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")

    with open(preset_path) as f:
        return yaml.safe_load(f)


def build_pipeline_from_recipe(recipe_dict: Dict[str, Any]) -> PreprocessPipeline:
    """Build preprocessing pipeline from recipe dictionary.

    Parameters
    ----------
    recipe_dict : Dict[str, Any]
        Recipe dictionary with 'steps' list.

    Returns
    -------
    PreprocessPipeline
        Constructed preprocessing pipeline.
    """
    steps: List[Step] = []
    recipe_steps = recipe_dict.get("steps", [])

    for step_cfg in recipe_steps:
        op_name = step_cfg.get("op")
        if not op_name:
            continue

        # Get operator class
        op_cls = _OPERATOR_REGISTRY.get(op_name)
        if op_cls is None:
            print(f"Warning: Unknown operator '{op_name}', skipping")
            continue

        # Extract parameters (everything except 'op')
        params = {k: v for k, v in step_cfg.items() if k != "op"}

        try:
            step = op_cls(**params)
            steps.append(step)
        except Exception as e:
            print(f"Error constructing operator '{op_name}': {e}")
            continue

    return PreprocessPipeline(steps=steps)


def load_recipe(
    preset: Optional[str] = None,
    protocol_config: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> PreprocessPipeline:
    """Load and merge preprocessing recipe.

    Priority: cli_overrides > protocol_config > preset > default

    Parameters
    ----------
    preset : str | None
        Preset name to load.
    protocol_config : Dict | None
        Protocol-level preprocessing configuration.
    cli_overrides : Dict | None
        CLI-level overrides.

    Returns
    -------
    PreprocessPipeline
        Constructed pipeline.
    """
    merged: Dict[str, Any] = {"modality": "unknown", "steps": []}

    # Load preset if specified
    if preset:
        preset_dict = load_preset_yaml(preset)
        merged.update(preset_dict)

    # Merge protocol config
    if protocol_config:
        if "preprocess" in protocol_config:
            pp_cfg = protocol_config["preprocess"]
            if isinstance(pp_cfg, dict):
                if "modality" in pp_cfg:
                    merged["modality"] = pp_cfg["modality"]
                if "steps" in pp_cfg:
                    merged["steps"] = pp_cfg["steps"]
            elif isinstance(pp_cfg, str):
                # Assume preset name
                preset_dict = load_preset_yaml(pp_cfg)
                merged.update(preset_dict)

    # Apply CLI overrides
    if cli_overrides:
        if "preset" in cli_overrides:
            preset_dict = load_preset_yaml(cli_overrides["preset"])
            merged.update(preset_dict)

        if "modality" in cli_overrides:
            merged["modality"] = cli_overrides["modality"]

        if "steps" in cli_overrides:
            merged["steps"] = cli_overrides["steps"]

        # Support step-level parameter overrides
        if "override_steps" in cli_overrides:
            for override in cli_overrides["override_steps"]:
                op_name = override.get("op")
                # Find and update step
                for i, step in enumerate(merged["steps"]):
                    if step.get("op") == op_name:
                        merged["steps"][i].update(override)
                        break

    return build_pipeline_from_recipe(merged)


def merge_recipe(*recipe_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple recipe dictionaries.

    Later recipes override earlier ones.

    Parameters
    ----------
    *recipe_dicts : Dict[str, Any]
        Recipe dictionaries to merge (in priority order).

    Returns
    -------
    Dict[str, Any]
        Merged recipe.
    """
    merged = {"modality": "unknown", "steps": []}

    for recipe in recipe_dicts:
        if "modality" in recipe and recipe["modality"] != "unknown":
            merged["modality"] = recipe["modality"]

        if "steps" in recipe:
            merged["steps"] = recipe["steps"]  # Replace, not append

    return merged


__all__ = [
    "list_operators",
    "load_preset_yaml",
    "build_pipeline_from_recipe",
    "load_recipe",
    "merge_recipe",
]
