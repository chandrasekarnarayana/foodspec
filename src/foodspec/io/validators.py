"""Validation helpers for FoodSpec IO."""
from __future__ import annotations

from typing import Dict, List

from foodspec.io.core import detect_format


def validate_input(path: str) -> Dict[str, List[str]]:
    """Validate input path and inferred format.

    Returns dict with "errors" and "warnings" lists.
    """

    errors: List[str] = []
    warnings: List[str] = []

    fmt = detect_format(path)
    if fmt == "unknown":
        errors.append("Unsupported or unknown input format.")
    elif fmt == "txt":
        warnings.append("Plain text inputs may require format hints.")

    return {"errors": errors, "warnings": warnings}


__all__ = ["validate_input"]

