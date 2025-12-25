"""Output bundle: unified management of workflow artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from foodspec.core.run_record import RunRecord


@dataclass
class OutputBundle:
    """Unified container for workflow outputs: metrics, diagnostics, provenance, artifacts.

    Manages the triple output (metrics + diagnostics + provenance) and exports to disk.

    Parameters
    ----------
    run_record : RunRecord
        Provenance record for the workflow.

    Attributes
    ----------
    metrics : dict
        Quantitative results (accuracy, F1, RMSE, etc.).
    diagnostics : dict
        Plots and tables (confusion matrix, feature importance, etc.).
    artifacts : dict
        Portable exports (model, preprocessor, etc.).
    run_record : RunRecord
        Provenance.
    """

    run_record: RunRecord
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    _output_dir: Optional[Path] = field(default=None, repr=False)

    def add_metrics(self, name: str, value: Any) -> None:
        """Add a metric.

        Parameters
        ----------
        name : str
            Metric name (e.g., "accuracy").
        value : Any
            Metric value (number, array, dataframe).
        """
        self.metrics[name] = value

    def add_diagnostic(self, name: str, value: Any) -> None:
        """Add a diagnostic (plot, table, figure).

        Parameters
        ----------
        name : str
            Diagnostic name (e.g., "confusion_matrix").
        value : Any
            Diagnostic (matplotlib Figure, np.ndarray, pd.DataFrame, dict).
        """
        self.diagnostics[name] = value

    def add_artifact(self, name: str, value: Any) -> None:
        """Add an artifact (model, preprocessor, scaler, etc.).

        Parameters
        ----------
        name : str
            Artifact name (e.g., "model").
        value : Any
            Artifact object.
        """
        self.artifacts[name] = value

    def export(
        self,
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
    ) -> Path:
        """Export bundle to disk.

        Exports:
        - metrics.json
        - diagnostics/ (plots as PNG/PDF, tables as CSV)
        - artifacts/ (models as joblib/pickle)
        - provenance.json (run_record)

        Parameters
        ----------
        output_dir : str or Path
            Output directory.
        formats : list of str, optional
            Export formats to use. Default: ['json', 'csv', 'png', 'joblib'].

        Returns
        -------
        Path
            Output directory path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir

        # Track export location in run record for reproducibility of artifacts.
        self.run_record.add_output_path(output_dir)

        if formats is None:
            formats = ["json", "csv", "png", "joblib"]

        # Export metrics
        self._export_metrics(output_dir, formats)

        # Export diagnostics
        self._export_diagnostics(output_dir, formats)

        # Export artifacts
        self._export_artifacts(output_dir, formats)

        # Export provenance
        self.run_record.to_json(output_dir / "provenance.json")

        return output_dir

    def _export_metrics(self, output_dir: Path, formats: List[str]) -> None:
        """Export metrics to disk."""
        if not self.metrics:
            return

        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        # JSON export
        if "json" in formats:
            metrics_serializable = self._make_serializable(self.metrics)
            (metrics_dir / "metrics.json").write_text(
                json.dumps(metrics_serializable, indent=2),
                encoding="utf-8"
            )

        # CSV export for DataFrames
        if "csv" in formats:
            for name, value in self.metrics.items():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(metrics_dir / f"{name}.csv", index=True)

    def _export_diagnostics(self, output_dir: Path, formats: List[str]) -> None:
        """Export diagnostics to disk."""
        if not self.diagnostics:
            return

        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)

        for name, value in self.diagnostics.items():
            # Matplotlib figures
            if hasattr(value, "savefig"):
                if "png" in formats:
                    value.savefig(diag_dir / f"{name}.png", dpi=150, bbox_inches="tight")
                if "pdf" in formats:
                    value.savefig(diag_dir / f"{name}.pdf", bbox_inches="tight")

            # DataFrames
            elif isinstance(value, pd.DataFrame):
                if "csv" in formats:
                    value.to_csv(diag_dir / f"{name}.csv", index=True)

            # NumPy arrays
            elif isinstance(value, np.ndarray):
                if "npy" in formats:
                    np.save(diag_dir / f"{name}.npy", value)

            # Dictionaries (serialize as JSON)
            elif isinstance(value, dict):
                if "json" in formats:
                    serializable = self._make_serializable(value)
                    (diag_dir / f"{name}.json").write_text(
                        json.dumps(serializable, indent=2),
                        encoding="utf-8"
                    )
            # Plain strings (write as HTML by default)
            elif isinstance(value, str):
                # Save as .html for dashboard-like diagnostics
                (diag_dir / f"{name}.html").write_text(value, encoding="utf-8")

    def _export_artifacts(self, output_dir: Path, formats: List[str]) -> None:
        """Export artifacts to disk."""
        if not self.artifacts:
            return

        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        for name, value in self.artifacts.items():
            if "joblib" in formats:
                import joblib
                joblib.dump(value, artifacts_dir / f"{name}.joblib")

            if "pickle" in formats:
                import pickle
                with open(artifacts_dir / f"{name}.pkl", "wb") as f:
                    pickle.dump(value, f)

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable types.

        Parameters
        ----------
        obj : Any
            Object to convert.

        Returns
        -------
        Any
            JSON-serializable version.
        """
        if isinstance(obj, dict):
            return {k: OutputBundle._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OutputBundle._make_serializable(v) for v in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def summary(self) -> str:
        """Generate summary string of outputs.

        Returns
        -------
        str
            Human-readable summary.
        """
        lines = [
            f"OutputBundle(run_id={self.run_record.run_id})",
            f"  Metrics: {len(self.metrics)} items",
            f"  Diagnostics: {len(self.diagnostics)} items",
            f"  Artifacts: {len(self.artifacts)} items",
        ]

        if self._output_dir:
            lines.append(f"  Exported to: {self._output_dir}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OutputBundle(run_id={self.run_record.run_id}, "
            f"metrics={len(self.metrics)}, diagnostics={len(self.diagnostics)}, "
            f"artifacts={len(self.artifacts)})"
        )
