"""
Quality control step for protocol execution.

Validates dataset structure, required columns, and data quality before
proceeding with analysis steps.
"""

from typing import Any, Dict

from .base import Step


class QCStep(Step):
    """Perform quality control checks on dataset."""

    name = "qc_checks"

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize QC step.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Step configuration with required_columns and class_col.
        """
        self.cfg = cfg

    def run(self, ctx: Dict[str, Any]):
        """Execute QC validation checks.

        Parameters
        ----------
        ctx : Dict[str, Any]
            Execution context with 'data' DataFrame.

        Raises
        ------
        ValueError
            If critical QC errors are detected.
        """
        from foodspec.modeling.validation.quality import validate_dataset

        df = ctx.get("data")
        required = self.cfg.get("required_columns", [])
        class_col = self.cfg.get("class_col")
        diag = validate_dataset(df, required_cols=required, class_col=class_col)
        ctx["logs"].append(f"[qc_checks] warnings={diag['warnings']}")
        if diag["errors"]:
            raise ValueError(f"QC failed: {diag['errors']}")
