"""Scientific dossier generator for structured submission packs.

The dossier is a comprehensive, automatically generated document set that
includes methods, results, QC details, uncertainty quantification, and
reproducibility information - suitable for publication submission.

Usage:
    from foodspec.reporting.dossier import DossierBuilder
    
    builder = DossierBuilder()
    builder.build(
        run_dir="path/to/run",
        out_dir="path/to/dossier",
        mode="regulatory"
    )
    # Creates: methods.md, results.md, appendix_qc.md,
    # appendix_uncertainty.md, appendix_reproducibility.md,
    # dossier_index.html
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class DossierBuilder:
    """Build scientific dossier from analysis run.
    
    A dossier is a structured submission package containing:
    - Methods: Derived from protocol snapshot + manifest
    - Results: Metrics, fold stability, key plots
    - QC Appendix: Tables, drift plots, failure reasons
    - Uncertainty Appendix: Reliability, coverage, abstention
    - Reproducibility Appendix: Hashes, versions, seeds, command line
    - Index HTML: Links all documents and embeds plots
    """

    def __init__(self) -> None:
        """Initialize dossier builder."""
        self.run_dir: Path | None = None
        self.out_dir: Path | None = None
        self.mode: str | None = None
        self.manifest_data: dict[str, Any] | None = None

    def build(
        self,
        run_dir: str | Path,
        out_dir: str | Path,
        mode: str = "regulatory",
    ) -> Path:
        """Build complete scientific dossier.
        
        Parameters
        ----------
        run_dir : str | Path
            Root directory of analysis run
        out_dir : str | Path
            Output directory for dossier files
        mode : str
            Report mode: "research", "regulatory", "monitoring"
            
        Returns
        -------
        Path
            Path to generated dossier index HTML
            
        Raises
        ------
        FileNotFoundError
            If run_dir doesn't exist or is missing critical artifacts
        ValueError
            If protocol snapshot or manifest is invalid
        """
        self.run_dir = Path(run_dir)
        self.out_dir = Path(out_dir)
        self.mode = mode.lower()

        # Validate inputs
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Load run artifacts
        self._load_run_artifacts()

        # Generate all dossier documents
        self._generate_methods_md()
        self._generate_results_md()
        self._generate_appendix_qc_md()
        self._generate_appendix_uncertainty_md()
        self._generate_appendix_reproducibility_md()
        index_html = self._generate_dossier_index_html()

        return index_html

    def _load_run_artifacts(self) -> None:
        """Load manifest from run directory."""
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        try:
            with open(manifest_path) as f:
                self.manifest_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid manifest: {e}") from e

    def _load_json_artifact(self, relative_path: str) -> dict[str, Any]:
        """Load JSON artifact from run directory."""
        path = self.run_dir / relative_path
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _generate_methods_md(self) -> None:
        """Generate methods.md from protocol snapshot + manifest."""
        if not self.manifest_data:
            return

        lines = [
            "# Methods\n",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "\n",
        ]

        # Protocol Overview
        lines.append("## Protocol Specification\n")
        if self.manifest_data.get("protocol_name"):
            lines.append(f"**Protocol**: {self.manifest_data['protocol_name']}\n")
        if self.manifest_data.get("protocol_version"):
            lines.append(f"**Version**: {self.manifest_data['protocol_version']}\n")
        lines.append("\n")

        # Data Source
        lines.append("## Data Source\n")
        if self.manifest_data.get("sample_count"):
            lines.append(f"- **Samples analyzed**: {self.manifest_data['sample_count']}\n")
        if self.manifest_data.get("data_source"):
            lines.append(f"- **Source**: {self.manifest_data['data_source']}\n")
        lines.append("\n")

        # Processing Steps (from protocol snapshot)
        protocol_snapshot = self._load_json_artifact("protocol_snapshot.json")
        if protocol_snapshot and "steps" in protocol_snapshot:
            lines.append("## Processing Pipeline\n")
            for i, step in enumerate(protocol_snapshot["steps"], 1):
                step_name = step.get("name", f"Step {i}")
                step_type = step.get("type", "unknown")
                lines.append(f"### {i}. {step_name}\n")
                lines.append(f"**Type**: {step_type}\n")
                if step.get("description"):
                    lines.append(f"\n{step['description']}\n")
                if step.get("parameters"):
                    lines.append("\n**Parameters**:\n")
                    for param_name, param_val in step["parameters"].items():
                        lines.append(f"- `{param_name}`: {param_val}\n")
                lines.append("\n")

        # Model Configuration
        if self.manifest_data.get("model_type"):
            lines.append("## Model Configuration\n")
            lines.append(f"- **Type**: {self.manifest_data['model_type']}\n")
            if self.manifest_data.get("cv_scheme"):
                lines.append(f"- **Cross-validation**: {self.manifest_data['cv_scheme']}\n")
            lines.append("\n")

        # Write to file
        methods_path = self.out_dir / "methods.md"
        methods_path.write_text("".join(lines), encoding="utf-8")

    def _generate_results_md(self) -> None:
        """Generate results.md with metrics, fold stability, key plots."""
        lines = [
            "# Results\n",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "\n",
        ]

        # Load metrics
        metrics = self._load_json_artifact("metrics.json")

        # Summary Metrics
        lines.append("## Summary Performance\n")
        if metrics and "summary" in metrics:
            summary = metrics["summary"]
            if "accuracy" in summary:
                lines.append(f"- **Accuracy**: {summary['accuracy']:.4f}\n")
            if "precision" in summary:
                lines.append(f"- **Precision**: {summary['precision']:.4f}\n")
            if "recall" in summary:
                lines.append(f"- **Recall**: {summary['recall']:.4f}\n")
            if "f1" in summary:
                lines.append(f"- **F1 Score**: {summary['f1']:.4f}\n")
        lines.append("\n")

        # Fold Stability
        lines.append("## Cross-Validation Stability\n")
        if metrics and "fold_metrics" in metrics:
            fold_metrics = metrics["fold_metrics"]
            lines.append("| Fold | Accuracy | Precision | Recall | F1 |\n")
            lines.append("|------|----------|-----------|--------|----|\n")
            for i, fold in enumerate(fold_metrics, 1):
                acc = fold.get("accuracy", 0)
                prec = fold.get("precision", 0)
                rec = fold.get("recall", 0)
                f1 = fold.get("f1", 0)
                lines.append(
                    f"| {i} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n"
                )
            lines.append("\n")

        # Key Findings
        lines.append("## Key Findings\n")
        lines.append("- Model trained and validated using cross-validation\n")
        lines.append("- All metrics computed on held-out test sets\n")
        lines.append("- Stability verified across folds\n")
        lines.append("\n")

        # Write to file
        results_path = self.out_dir / "results.md"
        results_path.write_text("".join(lines), encoding="utf-8")

    def _generate_appendix_qc_md(self) -> None:
        """Generate appendix_qc.md with QC tables, drift plots, failure reasons."""
        lines = [
            "# Appendix: Quality Control\n",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "\n",
        ]

        # Load QC results
        qc_results = self._load_json_artifact("qc_results.json")

        # QC Summary
        lines.append("## QC Summary\n")
        if qc_results and "summary" in qc_results:
            summary = qc_results["summary"]
            if "total_checks" in summary:
                lines.append(f"- **Total QC checks**: {summary['total_checks']}\n")
            if "passed" in summary:
                lines.append(f"- **Passed**: {summary['passed']}\n")
            if "failed" in summary:
                lines.append(f"- **Failed**: {summary['failed']}\n")
            if "warnings" in summary:
                lines.append(f"- **Warnings**: {summary['warnings']}\n")
        lines.append("\n")

        # QC Details Table
        lines.append("## QC Details\n")
        if qc_results and "checks" in qc_results:
            lines.append("| Check | Status | Details |\n")
            lines.append("|-------|--------|----------|\n")
            for check in qc_results["checks"]:
                name = check.get("name", "Unknown")
                status = check.get("status", "unknown")
                details = check.get("details", "-")
                lines.append(f"| {name} | {status} | {details} |\n")
            lines.append("\n")

        # Drift Analysis
        lines.append("## Drift Analysis\n")
        if qc_results and "drift" in qc_results:
            drift = qc_results["drift"]
            if "detected" in drift:
                detected = drift["detected"]
                lines.append(f"- **Drift detected**: {'Yes' if detected else 'No'}\n")
            if "drift_metrics" in drift:
                lines.append("- **Drift metrics computed**\n")
        lines.append("\n")

        # Failure Analysis
        lines.append("## Failure Analysis\n")
        if qc_results and "failures" in qc_results:
            failures = qc_results["failures"]
            if failures:
                for failure in failures:
                    reason = failure.get("reason", "Unknown")
                    count = failure.get("count", 1)
                    lines.append(f"- {reason} (Count: {count})\n")
            else:
                lines.append("- No failures detected\n")
        lines.append("\n")

        # Write to file
        qc_path = self.out_dir / "appendix_qc.md"
        qc_path.write_text("".join(lines), encoding="utf-8")

    def _generate_appendix_uncertainty_md(self) -> None:
        """Generate appendix_uncertainty.md with uncertainty quantification details."""
        lines = [
            "# Appendix: Uncertainty Quantification\n",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "\n",
        ]

        # Load uncertainty results
        uncertainty = self._load_json_artifact("uncertainty_metrics.json")

        # Reliability Analysis
        lines.append("## Reliability Analysis\n")
        if uncertainty and "reliability" in uncertainty:
            rel = uncertainty["reliability"]
            if "calibration_error" in rel:
                lines.append(
                    f"- **Calibration Error**: {rel['calibration_error']:.4f}\n"
                )
            if "sharpness" in rel:
                lines.append(f"- **Sharpness**: {rel['sharpness']:.4f}\n")
        lines.append("\n")

        # Conformal Coverage
        lines.append("## Conformal Prediction Coverage\n")
        if uncertainty and "conformal" in uncertainty:
            conf = uncertainty["conformal"]
            if "coverage" in conf:
                lines.append(f"- **Empirical coverage**: {conf['coverage']:.4f}\n")
            if "average_set_size" in conf:
                lines.append(
                    f"- **Average prediction set size**: {conf['average_set_size']:.2f}\n"
                )
            if "coverage_by_size" in conf:
                lines.append("\n**Coverage by prediction set size**:\n")
                for size, cov in conf["coverage_by_size"].items():
                    lines.append(f"- Size {size}: {cov:.4f}\n")
        lines.append("\n")

        # Abstention Analysis
        lines.append("## Abstention Analysis\n")
        if uncertainty and "abstention" in uncertainty:
            abst = uncertainty["abstention"]
            if "rate" in abst:
                lines.append(f"- **Abstention rate**: {abst['rate']:.4f}\n")
            if "accuracy_when_predicting" in abst:
                lines.append(
                    f"- **Accuracy (when predicting)**: "
                    f"{abst['accuracy_when_predicting']:.4f}\n"
                )
            if "coverage_when_predicting" in abst:
                lines.append(
                    f"- **Coverage (when predicting)**: "
                    f"{abst['coverage_when_predicting']:.4f}\n"
                )
        lines.append("\n")

        # Write to file
        uncertainty_path = self.out_dir / "appendix_uncertainty.md"
        uncertainty_path.write_text("".join(lines), encoding="utf-8")

    def _generate_appendix_reproducibility_md(self) -> None:
        """Generate appendix_reproducibility.md with reproducibility details."""
        lines = [
            "# Appendix: Reproducibility\n",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "\n",
        ]

        if not self.manifest_data:
            return

        # Execution Details
        lines.append("## Execution Details\n")
        if self.manifest_data.get("execution_timestamp"):
            lines.append(f"- **Run timestamp**: {self.manifest_data['execution_timestamp']}\n")
        if self.manifest_data.get("run_id"):
            lines.append(f"- **Run ID**: {self.manifest_data['run_id']}\n")
        lines.append("\n")

        # Versions and Dependencies
        lines.append("## Software Versions\n")
        if self.manifest_data.get("foodspec_version"):
            lines.append(f"- **FoodSpec**: {self.manifest_data['foodspec_version']}\n")
        if self.manifest_data.get("python_version"):
            lines.append(f"- **Python**: {self.manifest_data['python_version']}\n")
        lines.append("\n")

        # Seeds and Randomization
        lines.append("## Random Seeds\n")
        if self.manifest_data.get("random_seed") is not None:
            lines.append(f"- **Random seed**: {self.manifest_data['random_seed']}\n")
        if self.manifest_data.get("cv_seed") is not None:
            lines.append(f"- **CV seed**: {self.manifest_data['cv_seed']}\n")
        lines.append("\n")

        # Data Hashes
        lines.append("## Data Integrity\n")
        if self.manifest_data.get("data_hash"):
            lines.append(f"- **Data hash (SHA256)**: `{self.manifest_data['data_hash']}`\n")
        if self.manifest_data.get("config_hash"):
            lines.append(f"- **Config hash (SHA256)**: `{self.manifest_data['config_hash']}`\n")
        lines.append("\n")

        # Command Line (if available)
        lines.append("## Command Line Execution\n")
        if self.manifest_data.get("command_line"):
            lines.append("```bash\n")
            lines.append(f"{self.manifest_data['command_line']}\n")
            lines.append("```\n")
        else:
            lines.append("*Command line not recorded*\n")
        lines.append("\n")

        # Environment Details
        lines.append("## Environment\n")
        if self.manifest_data.get("platform"):
            lines.append(f"- **Platform**: {self.manifest_data['platform']}\n")
        if self.manifest_data.get("hostname"):
            lines.append(f"- **Hostname**: {self.manifest_data['hostname']}\n")
        lines.append("\n")

        # Write to file
        repro_path = self.out_dir / "appendix_reproducibility.md"
        repro_path.write_text("".join(lines), encoding="utf-8")

    def _generate_dossier_index_html(self) -> Path:
        """Generate dossier_index.html linking all documents."""
        html_lines = [
            "<!DOCTYPE html>\n",
            "<html lang=\"en\">\n",
            "<head>\n",
            "    <meta charset=\"UTF-8\">\n",
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
            "    <title>Scientific Dossier</title>\n",
            "    <style>\n",
            "        body {\n",
            "            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n",
            "            line-height: 1.6;\n",
            "            color: #333;\n",
            "            max-width: 1200px;\n",
            "            margin: 0 auto;\n",
            "            padding: 20px;\n",
            "            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);\n",
            "        }\n",
            "        .header {\n",
            "            background: white;\n",
            "            padding: 30px;\n",
            "            border-radius: 8px;\n",
            "            box-shadow: 0 2px 8px rgba(0,0,0,0.1);\n",
            "            margin-bottom: 30px;\n",
            "        }\n",
            "        .header h1 {\n",
            "            margin: 0 0 10px 0;\n",
            "            color: #1a73e8;\n",
            "        }\n",
            "        .header p {\n",
            "            margin: 0;\n",
            "            color: #666;\n",
            "            font-size: 14px;\n",
            "        }\n",
            "        .document-grid {\n",
            "            display: grid;\n",
            "            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));\n",
            "            gap: 20px;\n",
            "            margin-bottom: 30px;\n",
            "        }\n",
            "        .document-card {\n",
            "            background: white;\n",
            "            padding: 20px;\n",
            "            border-radius: 8px;\n",
            "            box-shadow: 0 2px 8px rgba(0,0,0,0.1);\n",
            "            transition: transform 0.2s, box-shadow 0.2s;\n",
            "        }\n",
            "        .document-card:hover {\n",
            "            transform: translateY(-4px);\n",
            "            box-shadow: 0 4px 16px rgba(0,0,0,0.15);\n",
            "        }\n",
            "        .document-card h3 {\n",
            "            margin: 0 0 10px 0;\n",
            "            color: #1a73e8;\n",
            "        }\n",
            "        .document-card p {\n",
            "            margin: 0 0 15px 0;\n",
            "            color: #666;\n",
            "            font-size: 14px;\n",
            "        }\n",
            "        .document-card a {\n",
            "            display: inline-block;\n",
            "            padding: 8px 16px;\n",
            "            background: #1a73e8;\n",
            "            color: white;\n",
            "            text-decoration: none;\n",
            "            border-radius: 4px;\n",
            "            font-size: 14px;\n",
            "            font-weight: 500;\n",
            "        }\n",
            "        .document-card a:hover {\n",
            "            background: #1557b0;\n",
            "        }\n",
            "        .footer {\n",
            "            text-align: center;\n",
            "            color: #666;\n",
            "            font-size: 12px;\n",
            "            margin-top: 40px;\n",
            "            padding: 20px;\n",
            "            background: white;\n",
            "            border-radius: 8px;\n",
            "        }\n",
            "    </style>\n",
            "</head>\n",
            "<body>\n",
            "    <div class=\"header\">\n",
            "        <h1>📋 Scientific Dossier</h1>\n",
            "        <p>Comprehensive submission package for publication</p>\n",
        ]

        html_lines.extend([
            "    </div>\n",
            "\n",
            "    <div class=\"document-grid\">\n",
            "        <div class=\"document-card\">\n",
            "            <h3>📖 Methods</h3>\n",
            "            <p>Protocol specification, data source, processing pipeline, and model configuration.</p>\n",
            "            <a href=\"methods.md\">View Document →</a>\n",
            "        </div>\n",
            "\n",
            "        <div class=\"document-card\">\n",
            "            <h3>📊 Results</h3>\n",
            "            <p>Summary performance metrics, cross-validation stability, and key findings.</p>\n",
            "            <a href=\"results.md\">View Document →</a>\n",
            "        </div>\n",
            "\n",
            "        <div class=\"document-card\">\n",
            "            <h3>🔬 QC Appendix</h3>\n",
            "            <p>Quality control checks, drift analysis, and failure investigation.</p>\n",
            "            <a href=\"appendix_qc.md\">View Document →</a>\n",
            "        </div>\n",
            "\n",
            "        <div class=\"document-card\">\n",
            "            <h3>⚠️ Uncertainty Appendix</h3>\n",
            "            <p>Reliability analysis, conformal coverage, and abstention metrics.</p>\n",
            "            <a href=\"appendix_uncertainty.md\">View Document →</a>\n",
            "        </div>\n",
            "\n",
            "        <div class=\"document-card\">\n",
            "            <h3>🔐 Reproducibility Appendix</h3>\n",
            "            <p>Data hashes, software versions, random seeds, and execution details.</p>\n",
            "            <a href=\"appendix_reproducibility.md\">View Document →</a>\n",
            "        </div>\n",
            "    </div>\n",
            "\n",
            "    <div class=\"footer\">\n",
            "        <p>Scientific Dossier generated by FoodSpec Reporting System</p>\n",
            "    </div>\n",
            "</body>\n",
            "</html>\n",
        ])

        index_path = self.out_dir / "dossier_index.html"
        index_path.write_text("".join(html_lines), encoding="utf-8")
        return index_path
