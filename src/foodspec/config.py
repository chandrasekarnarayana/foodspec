"""Simple configuration loader/merger for foodspec CLI."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict

__all__ = ["load_config", "merge_cli_overrides"]


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a config file (YAML or JSON) into a dict."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with `pip install pyyaml`."
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Config extension must be .yml/.yaml or .json.")


def merge_cli_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI overrides (non-None) into base config."""

    merged = dict(config)
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged
