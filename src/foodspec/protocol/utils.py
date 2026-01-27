"""
Protocol utility functions.

Provides functions for discovering, loading, and validating protocol files.
"""

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from foodspec.modeling.outcome import OutcomeType

from .config import ProtocolConfig
from .steps import STEP_REGISTRY


def list_available_protocols(proto_dir: Union[str, Path] = "examples/protocols") -> List[Path]:
    """List all available protocol files in a directory.

    Parameters
    ----------
    proto_dir : Union[str, Path], optional
        Directory to search for protocol files, by default "examples/protocols"

    Returns
    -------
    List[Path]
        List of paths to protocol files (.yml, .yaml, .json).
    """
    p = Path(proto_dir)
    if not p.exists():
        return []
    return list(p.glob("*.yml")) + list(p.glob("*.yaml")) + list(p.glob("*.json"))


def load_protocol(name: str, proto_dir: Union[str, Path] = "examples/protocols") -> ProtocolConfig:
    """Load a protocol by name from a directory.

    Parameters
    ----------
    name : str
        Protocol filename (with or without extension).
    proto_dir : Union[str, Path], optional
        Directory containing protocol files, by default "examples/protocols"

    Returns
    -------
    ProtocolConfig
        Loaded protocol configuration.

    Raises
    ------
    FileNotFoundError
        If protocol file not found.
    """
    p = Path(proto_dir)
    path = p / name
    if not path.exists():
        # try add extension
        for ext in [".yaml", ".yml", ".json"]:
            if (p / (name + ext)).exists():
                path = p / (name + ext)
                break
    if not path.exists():
        raise FileNotFoundError(f"Protocol {name} not found in {proto_dir}")
    return ProtocolConfig.from_file(path)


def validate_protocol(cfg: ProtocolConfig, df: pd.DataFrame) -> Dict[str, List[str]]:
    """Validate protocol configuration against a dataset.

    Parameters
    ----------
    cfg : ProtocolConfig
        Protocol configuration to validate.
    df : pd.DataFrame
        Dataset to validate against.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'errors' and 'warnings' lists.
    """
    errors: List[str] = []
    warnings: List[str] = []
    if not cfg.name:
        errors.append("Protocol name missing.")
    if not cfg.version:
        warnings.append("Protocol version missing; defaulting.")
    if not cfg.steps:
        errors.append("Protocol has no steps.")
    if cfg.min_foodspec_version:
        # Best effort version check
        try:
            from foodspec import __version__ as fs_version

            if fs_version < cfg.min_foodspec_version:
                warnings.append(f"Protocol expects FoodSpec >= {cfg.min_foodspec_version}, running {fs_version}.")
        except Exception:
            warnings.append("Could not verify FoodSpec version.")
    # step type validation
    for step in cfg.steps:
        if step.get("type") not in STEP_REGISTRY:
            errors.append(f"Unknown step type: {step.get('type')}")
    # outcome typing
    outcome = (cfg.outcome_type or "classification").lower()
    target_col = cfg.target_column or cfg.expected_columns.get("target_column")
    if outcome in {OutcomeType.REGRESSION.value, OutcomeType.COUNT.value, OutcomeType.CLASSIFICATION.value}:
        if not target_col:
            errors.append("target_column is required for modeling tasks; set task.target_column in protocol.")
        elif target_col not in df.columns:
            errors.append(f"target_column '{target_col}' not found in dataset.")
    if outcome == OutcomeType.SURVIVAL.value:
        if not target_col or target_col not in df.columns:
            errors.append("Survival tasks require target_column (event indicator) present in data.")
        if not cfg.time_column or cfg.time_column not in df.columns:
            errors.append("Survival tasks require time_column present in data.")
    if outcome == OutcomeType.COUNT.value:
        if target_col and target_col in df.columns:
            if (df[target_col] < 0).any():
                errors.append(f"Count outcome '{target_col}' must be non-negative.")

    # expected columns
    if cfg.expected_columns:
        for _, col in cfg.expected_columns.items():
            if col and col not in df.columns:
                errors.append(
                    f"Required column '{col}' not found. Map columns correctly or adjust protocol expected_columns."
                )
    if cfg.min_foodspec_version:
        try:
            from foodspec import __version__ as fs_version

            warnings.append(
                f"Protocol declares min foodspec {cfg.min_foodspec_version}; running on {fs_version}. Ensure compatibility."
            )
        except Exception:
            warnings.append("Could not determine FoodSpec version for protocol compatibility check.")
    if cfg.required_metadata:
        for m in cfg.required_metadata:
            if m not in df.columns:
                warnings.append(f"Required metadata '{m}' not found in dataset.")
    # class count
    oil_col = cfg.expected_columns.get("oil_col")
    if outcome == OutcomeType.CLASSIFICATION.value and oil_col and oil_col in df.columns:
        if df[oil_col].nunique(dropna=True) < 2:
            errors.append("Only one class present; add more classes/samples before running discrimination.")
        min_count = df[oil_col].value_counts(dropna=True).min()
        if pd.notna(min_count) and min_count < 2:
            warnings.append(
                f"Very small class count detected (min {min_count}); collect more samples or adjust protocol."
            )
        elif pd.notna(min_count) and min_count < 3:
            warnings.append(f"Small class count (min {min_count}); CV folds will be reduced automatically.")
    # constant columns
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            warnings.append(f"Column '{col}' is constant.")
    # feature/samples ratio
    num_features = df.select_dtypes(include=["number"]).shape[1]
    num_samples = len(df)
    if num_samples and num_features > 10 * num_samples:
        warnings.append(
            f"High feature-to-sample ratio ({num_features} features vs {num_samples} samples); "
            "consider feature capping or simpler normalization."
        )
    return {"errors": errors, "warnings": warnings}
