"""Metadata validation helpers for FoodSpec datasets."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from foodspec.core.summary import summarize_dataset
from foodspec.utils.troubleshooting import check_missing_metadata


def validate_metadata(metadata: pd.DataFrame, label_column: str | None = None) -> Dict[str, List[str]]:
    """Run lightweight metadata checks for required fields and class balance.

    Args:
        metadata: Sample metadata table.
        label_column: Optional label column for balance checks.

    Returns:
        Dict with keys "errors" and "warnings" listing validation messages.
    """

    errors: List[str] = []
    warnings: List[str] = []

    if metadata is None or metadata.empty:
        errors.append("Metadata table is empty.")
        return {"errors": errors, "warnings": warnings}

    missing = check_missing_metadata(metadata)
    if missing:
        warnings.append(f"Missing metadata fields: {', '.join(missing)}")

    if label_column:
        try:
            summary = summarize_dataset(metadata, label_column=label_column)
            if summary.get("imbalance_ratio", 1) > 10:
                warnings.append("Severe class imbalance detected in metadata.")
        except Exception as exc:
            warnings.append(f"Metadata balance check failed: {exc}")

    return {"errors": errors, "warnings": warnings}


__all__ = ["validate_metadata"]
