"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

ArtifactRegistry manages standard run artifacts under a run directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class ArtifactRegistry:
    """Manage standard artifact locations and safe writes.

    Standard layout (within ``root``)::
        metrics.csv
        qc.csv
        predictions.csv
        plots/
        report.html
        report.pdf
        bundle/
        manifest.json
        logs.txt

    Examples
    --------
    Create a run layout and write metrics::

        reg = ArtifactRegistry(Path("/tmp/run"))
        reg.ensure_layout()
        reg.write_csv(reg.metrics_path, [{"metric": "accuracy", "value": 0.91}])
        reg.write_json(reg.manifest_path, {"version": "2.0.0"})
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    # Path helpers
    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.csv"

    @property
    def metrics_per_fold_path(self) -> Path:
        return self.root / "metrics_per_fold.csv"

    @property
    def metrics_summary_path(self) -> Path:
        return self.root / "metrics_summary.csv"

    @property
    def best_params_path(self) -> Path:
        return self.root / "best_params.csv"

    @property
    def qc_path(self) -> Path:
        return self.root / "qc.csv"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.csv"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    @property
    def report_html_path(self) -> Path:
        return self.root / "report.html"

    @property
    def report_pdf_path(self) -> Path:
        return self.root / "report.pdf"

    @property
    def bundle_dir(self) -> Path:
        return self.root / "bundle"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs_path(self) -> Path:
        return self.root / "logs.txt"

    @property
    def trust_dir(self) -> Path:
        """Directory for trust and uncertainty artifacts."""
        return self.root / "trust"

    @property
    def trust_eval_path(self) -> Path:
        """Trust evaluation result file."""
        return self.trust_dir / "evaluation.json"

    @property
    def prediction_sets_path(self) -> Path:
        """Conformal prediction sets."""
        return self.trust_dir / "prediction_sets.csv"

    @property
    def abstention_path(self) -> Path:
        """Abstention summary."""
        return self.trust_dir / "abstention.csv"

    @property
    def coverage_table_path(self) -> Path:
        """Per-group coverage table."""
        return self.trust_dir / "coverage_table.csv"

    @property
    def calibration_path(self) -> Path:
        """Calibration artifacts and parameters."""
        return self.trust_dir / "calibration.json"

    @property
    def calibration_metrics_path(self) -> Path:
        """Calibration metrics (ECE, MCE, etc.)."""
        return self.trust_dir / "calibration_metrics.csv"

    @property
    def conformal_coverage_path(self) -> Path:
        """Overall conformal coverage aggregated across folds."""
        return self.trust_dir / "conformal_coverage.csv"

    @property
    def conformal_sets_path(self) -> Path:
        """Conformal prediction sets per sample."""
        return self.trust_dir / "conformal_sets.csv"

    @property
    def abstention_summary_path(self) -> Path:
        """Abstention rule summary and rates."""
        return self.trust_dir / "abstention_summary.csv"

    @property
    def coefficients_path(self) -> Path:
        """Model coefficients from interpretability analysis."""
        return self.trust_dir / "coefficients.csv"

    @property
    def permutation_importance_path(self) -> Path:
        """Permutation importance analysis."""
        return self.trust_dir / "permutation_importance.csv"

    @property
    def marker_panel_explanations_path(self) -> Path:
        """Marker panel aligned with interpretability outputs."""
        return self.trust_dir / "marker_panel_explanations.csv"

    def ensure_layout(self) -> None:
        """Create root and standard directories if missing."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.trust_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(self, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        """Write rows to CSV with headers inferred from first row."""

        self._ensure_parent(path)
        rows = list(rows)
        if not rows:
            path.write_text("")
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Write a mapping to JSON (pretty-printed)."""

        self._ensure_parent(path)
        path.write_text(json.dumps(payload, indent=2))

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def write_trust_calibration_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Write calibration metrics (ECE, MCE) to CSV.
        
        Parameters
        ----------
        metrics : Mapping[str, Any]
            Dictionary with metric names as keys (e.g., 'ece', 'mce')
        """
        rows = [metrics]
        self.write_csv(self.calibration_metrics_path, rows)

    def write_trust_coverage(self, coverage_df: Any) -> None:
        """Write conformal coverage table.
        
        Parameters
        ----------
        coverage_df : pd.DataFrame
            Coverage table from coverage_by_group() or aggregated coverage
        """
        self.write_csv(self.conformal_coverage_path, coverage_df.to_dict(orient='records'))

    def write_trust_conformal_sets(self, conformal_df: Any) -> None:
        """Write conformal prediction sets.
        
        Parameters
        ----------
        conformal_df : pd.DataFrame
            Conformal sets output from ConformalPredictionResult.to_dataframe()
        """
        self.write_csv(self.conformal_sets_path, conformal_df.to_dict(orient='records'))

    def write_trust_abstention_summary(self, abstention_summary: Mapping[str, Any]) -> None:
        """Write abstention rule summary.
        
        Parameters
        ----------
        abstention_summary : Mapping[str, Any]
            Abstention metrics (abstain_rate, accuracy_on_answered, etc.)
        """
        rows = [abstention_summary]
        self.write_csv(self.abstention_summary_path, rows)

    def write_trust_coefficients(self, coef_df: Any) -> None:
        """Write model coefficients from interpretability analysis.
        
        Parameters
        ----------
        coef_df : pd.DataFrame
            Output from extract_linear_coefficients()
        """
        self.write_csv(self.coefficients_path, coef_df.to_dict(orient='records'))

    def write_trust_permutation_importance(self, importance_df: Any) -> None:
        """Write permutation importance analysis.
        
        Parameters
        ----------
        importance_df : pd.DataFrame
            Output from permutation_importance_with_names()
        """
        self.write_csv(self.permutation_importance_path, importance_df.to_dict(orient='records'))

    def write_trust_marker_panel_explanations(self, explanations_df: Any) -> None:
        """Write marker panel explanations aligned with interpretability.
        
        Parameters
        ----------
        explanations_df : pd.DataFrame
            Output from link_marker_panel_explanations()
        """
        self.write_csv(self.marker_panel_explanations_path, explanations_df.to_dict(orient='records'))


__all__ = ["ArtifactRegistry"]
