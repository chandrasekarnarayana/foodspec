"""
Reporting engine that assembles FoodSpec artifacts into deterministic reports.

Key guarantees:
- One run -> one complete report bundle (HTML first, optional PDF stub).
- Consumes ArtifactRegistry outputs and RunManifest; no recompute unless allowed.
- Supports reporting modes: research, regulatory, monitoring.
- Produces experiment card, scientific dossier, and exportable archive.
- Provides run comparison utilities across multiple run folders.
- High-DPI, deterministic filenames.
"""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.manifest import RunManifest
from foodspec.reporting.report import DEFAULT_TEMPLATE, generate_html_report
from foodspec.reporting.modes import ReportMode, get_mode_config
from foodspec.viz.paper_styles import apply_paper_style, DEFAULT_DPI


@dataclass
class ReportOutputs:
    html: Path
    experiment_card_html: Path
    experiment_card_json: Path
    dossier_html: Path
    archive_path: Optional[Path]
    comparison_html: Optional[Path] = None


class ReportingEngine:
    """Build reports from existing artifacts and a manifest."""

    def __init__(
        self,
        artifacts: ArtifactRegistry,
        manifest: RunManifest | Mapping[str, Any],
        mode: ReportMode | str = ReportMode.RESEARCH,
        allow_recompute: bool = False,
    ) -> None:
        self.artifacts = artifacts
        self.mode = ReportMode(mode) if isinstance(mode, str) else mode
        self.mode_config = get_mode_config(self.mode)
        self.allow_recompute = allow_recompute
        if isinstance(manifest, RunManifest):
            self.manifest = manifest
            self.manifest_payload = manifest.__dict__
        else:
            self.manifest = None
            self.manifest_payload = dict(manifest)

    # -------- Public API --------
    def build(self) -> ReportOutputs:
        self.artifacts.ensure_layout()
        run_data = self._load_run_data()

        html_path = self._render_main_report(run_data)
        card_html, card_json = self._render_experiment_card(run_data)
        dossier_html = self._render_dossier(run_data)
        archive_path = self._build_archive([html_path, card_html, dossier_html])

        return ReportOutputs(
            html=html_path,
            experiment_card_html=card_html,
            experiment_card_json=card_json,
            dossier_html=dossier_html,
            archive_path=archive_path,
        )

    # -------- Data loading (no recompute) --------
    def _load_run_data(self) -> Dict[str, Any]:
        manifest = self.manifest_payload
        protocol = manifest.get("protocol_snapshot", {}) if manifest else {}

        metrics = self._read_csv(self.artifacts.metrics_summary_path) or self._read_csv(self.artifacts.metrics_path)
        qc_rows = self._read_csv(self.artifacts.qc_path)
        uncertainty = self._read_json_if_exists(self.artifacts.trust_eval_path) or {}
        calibration_metrics = self._read_csv(self.artifacts.calibration_metrics_path)

        plots = self._collect_plots()

        return {
            "manifest": manifest,
            "protocol": protocol,
            "metrics": metrics,
            "qc": qc_rows,
            "uncertainty": uncertainty,
            "calibration_metrics": calibration_metrics,
            "plots": plots,
            "mode": self.mode.value,
        }

    def _collect_plots(self) -> List[str]:
        plot_paths: List[Path] = []
        if self.artifacts.plots_dir.exists():
            plot_paths.extend(sorted(self.artifacts.plots_dir.glob("**/*.png")))
            plot_paths.extend(sorted(self.artifacts.plots_dir.glob("**/*.svg")))
        # Limit to relative paths for HTML portability
        return [str(p.relative_to(self.artifacts.root)) if p.is_absolute() else str(p) for p in plot_paths]

    # -------- Rendering helpers --------
    def _render_main_report(self, run_data: Mapping[str, Any]) -> Path:
        manifest = run_data["manifest"]
        protocol = run_data.get("protocol", {})

        dataset_summary = self._build_dataset_summary(protocol)
        preprocessing_steps = self._build_preprocessing_steps(protocol)

        generate_html_report(
            artifacts=self.artifacts,
            manifest=manifest,
            dataset_summary=dataset_summary,
            preprocessing_steps=preprocessing_steps,
            qc_table=run_data.get("qc", []),
            metrics=run_data.get("metrics", []),
            plots=run_data.get("plots", []),
            uncertainty=run_data.get("uncertainty", {}),
            template_str=DEFAULT_TEMPLATE,
        )

        self._write_pdf_stub()
        return self.artifacts.report_html_path

    def _render_experiment_card(self, run_data: Mapping[str, Any]) -> tuple[Path, Path]:
        manifest = run_data["manifest"]
        metrics = run_data.get("metrics", [])

        card = {
            "mode": self.mode.value,
            "seed": manifest.get("seed"),
            "data_fingerprint": manifest.get("data_fingerprint"),
            "protocol_hash": manifest.get("protocol_hash"),
            "headline_metric": self._extract_headline_metric(metrics),
            "risks": self._extract_risks(run_data),
            "confidence": self._extract_confidence(run_data),
            "deployment_readiness": self._deployment_readiness(run_data),
        }

        card_json_path = self.artifacts.experiment_card_json_path
        card_json_path.parent.mkdir(parents=True, exist_ok=True)
        card_json_path.write_text(json.dumps(card, indent=2))

        card_html = self.artifacts.experiment_card_path
        card_html.parent.mkdir(parents=True, exist_ok=True)
        card_html.write_text(self._render_card_html(card))
        return card_html, card_json_path

    def _render_dossier(self, run_data: Mapping[str, Any]) -> Path:
        manifest = run_data["manifest"]
        sections = self.mode_config.enabled_sections

        dossier_html = self.artifacts.dossier_html_path
        dossier_html.parent.mkdir(parents=True, exist_ok=True)

        html_parts = [
            "<html><head><meta charset='UTF-8'><title>FoodSpec Scientific Dossier</title></head><body>",
            f"<h1>Scientific Dossier ({self.mode.value.title()})</h1>",
            f"<p>Protocol hash: {manifest.get('protocol_hash', '')}</p>",
            f"<p>Data fingerprint: {manifest.get('data_fingerprint', '')}</p>",
        ]

        if "summary" in sections:
            html_parts.append("<h2>Run Summary</h2>")
            html_parts.append(self._summary_list(manifest))

        if "dataset" in sections:
            html_parts.append("<h2>Methods</h2>")
            html_parts.append(self._methods_section(run_data.get("protocol", {})))

        if "metrics" in sections:
            html_parts.append("<h2>Results</h2>")
            html_parts.append(self._table_html(run_data.get("metrics", [])))

        if "qc" in sections:
            html_parts.append("<h2>QC Appendix</h2>")
            html_parts.append(self._table_html(run_data.get("qc", [])))

        if "uncertainty" in sections:
            html_parts.append("<h2>Uncertainty Appendix</h2>")
            html_parts.append(self._uncertainty_section(run_data))

        html_parts.append("<h2>Reproducibility Package</h2>")
        html_parts.append("<p>See manifest.json and metrics in bundle.</p>")

        html_parts.append("</body></html>")
        dossier_html.write_text("\n".join(html_parts))
        return dossier_html

    def _build_archive(self, files: Iterable[Path]) -> Path:
        archive_path = self.artifacts.report_bundle_path
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                if path.exists():
                    zf.write(path, path.relative_to(self.artifacts.root))
            # Always include manifest if present
            manifest_path = self.artifacts.manifest_path
            if manifest_path.exists():
                zf.write(manifest_path, manifest_path.relative_to(self.artifacts.root))
        return archive_path

    # -------- Run comparison --------
    def compare_runs(
        self,
        run_dirs: Sequence[Path],
        output_dir: Optional[Path] = None,
        style: str = "joss",
    ) -> Path:
        """Generate run comparison dashboard (HTML) with radar plot and leaderboard."""

        out_dir = output_dir or self.artifacts.comparison_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        run_summaries = [self._load_run_summary_from_dir(r) for r in run_dirs]
        leaderboard_path = out_dir / "leaderboard.csv"
        self._write_leaderboard(run_summaries, leaderboard_path)

        radar_path = out_dir / "run_comparison_radar.png"
        self._make_radar_plot(run_summaries, radar_path, style)

        html_path = out_dir / "run_comparison.html"
        html_path.write_text(self._render_comparison_html(run_summaries, leaderboard_path, radar_path))
        return html_path

    # -------- Utility methods --------
    def _read_csv(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, Any]] = []
            for row in reader:
                rows.append({k: self._maybe_float(v) for k, v in row.items()})
            return rows

    def _read_json_if_exists(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    @staticmethod
    def _maybe_float(value: str) -> Any:
        try:
            return float(value) if value not in {"", None} else value
        except Exception:
            return value

    @staticmethod
    def _build_dataset_summary(protocol: Mapping[str, Any]) -> Dict[str, Any]:
        data = protocol.get("data", {}) if protocol else {}
        return {
            "modality": data.get("modality", ""),
            "input": data.get("input", ""),
            "label": data.get("label", ""),
        }

    @staticmethod
    def _build_preprocessing_steps(protocol: Mapping[str, Any]) -> List[str]:
        preprocess = protocol.get("preprocess", {}) if protocol else {}
        steps = preprocess.get("steps", []) or preprocess.get("recipe", [])
        if isinstance(steps, list):
            return [str(s) for s in steps]
        return [str(steps)] if steps else []

    @staticmethod
    def _extract_headline_metric(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        row = metrics[0]
        for key, val in row.items():
            if key.lower() in {"fold_id", "fold"}:
                continue
            return {"name": key, "value": val}
        return {}

    @staticmethod
    def _extract_risks(run_data: Mapping[str, Any]) -> List[str]:
        risks = []
        manifest = run_data.get("manifest", {})
        if manifest.get("warnings"):
            risks.extend(manifest["warnings"])
        if not run_data.get("qc"):
            risks.append("QC table missing")
        return risks

    @staticmethod
    def _extract_confidence(run_data: Mapping[str, Any]) -> Dict[str, Any]:
        calibration = run_data.get("calibration_metrics", [])
        if calibration:
            row = calibration[0]
            return {k: v for k, v in row.items() if isinstance(v, (int, float))}
        return {}

    @staticmethod
    def _deployment_readiness(run_data: Mapping[str, Any]) -> str:
        if run_data.get("uncertainty"):
            return "ready_with_uncertainty"
        if run_data.get("qc"):
            return "ready_pending_uncertainty"
        return "needs_qc"

    @staticmethod
    def _render_card_html(card: Mapping[str, Any]) -> str:
        return """
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Experiment Card</title></head>
<body>
  <h1>Experiment Card</h1>
  <p>Mode: {mode}</p>
  <p>Protocol hash: {protocol_hash}</p>
  <p>Data fingerprint: {data_fingerprint}</p>
  <p>Headline metric: {headline}</p>
  <p>Deployment readiness: {readiness}</p>
</body></html>
""".format(
            mode=card.get("mode", ""),
            protocol_hash=card.get("protocol_hash", ""),
            data_fingerprint=card.get("data_fingerprint", ""),
            headline=json.dumps(card.get("headline_metric", {})),
            readiness=card.get("deployment_readiness", ""),
        )

    @staticmethod
    def _table_html(rows: Sequence[Mapping[str, Any]]) -> str:
        if not rows:
            return "<p>No data.</p>"
        cols = list(rows[0].keys())
        header = "".join(f"<th>{c}</th>" for c in cols)
        body_rows = []
        for row in rows:
            body_rows.append("<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in cols) + "</tr>")
        return "<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(header, "".join(body_rows))

    def _uncertainty_section(self, run_data: Mapping[str, Any]) -> str:
        parts = []
        if run_data.get("calibration_metrics"):
            parts.append("<h3>Calibration Metrics</h3>")
            parts.append(self._table_html(run_data["calibration_metrics"]))
        if run_data.get("uncertainty"):
            parts.append("<pre>" + json.dumps(run_data["uncertainty"], indent=2) + "</pre>")
        return "".join(parts) if parts else "<p>No uncertainty artifacts.</p>"

    @staticmethod
    def _methods_section(protocol: Mapping[str, Any]) -> str:
        data = protocol.get("data", {}) if protocol else {}
        preprocess = protocol.get("preprocess", {}) if protocol else {}
        model = protocol.get("model", {}) if protocol else {}
        items = [
            ("Modality", data.get("modality", "")),
            ("Input", data.get("input", "")),
            ("Preprocess", preprocess.get("recipe", "")),
            ("Model", model.get("estimator", "")),
        ]
        return "<ul>" + "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in items) + "</ul>"

    @staticmethod
    def _summary_list(manifest: Mapping[str, Any]) -> str:
        items = [
            ("Version", manifest.get("protocol_snapshot", {}).get("version", "")),
            ("Seed", manifest.get("seed", "")),
            ("Start", manifest.get("start_time", "")),
            ("End", manifest.get("end_time", "")),
            ("Duration", manifest.get("duration_seconds", "")),
        ]
        return "<ul>" + "".join(f"<li><strong>{k}:</strong> {v}</li>" for k, v in items) + "</ul>"

    def _write_pdf_stub(self) -> None:
        pdf_path = self.artifacts.report_pdf_path
        if pdf_path.exists():
            return
        pdf_path.write_text("PDF export not generated (optional dependency).")

    def _load_run_summary_from_dir(self, run_dir: Path) -> Dict[str, Any]:
        reg = ArtifactRegistry(run_dir)
        manifest_path = reg.manifest_path
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        metrics = []
        for candidate in [reg.metrics_summary_path, reg.metrics_path]:
            metrics = self._read_csv(candidate)
            if metrics:
                break
        return {
            "run_dir": run_dir,
            "name": run_dir.name,
            "manifest": manifest,
            "metrics": metrics,
        }

    def _write_leaderboard(self, runs: Sequence[Mapping[str, Any]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not runs:
            output_path.write_text("")
            return
        metric_keys = self._metric_keys(runs)
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", *metric_keys])
            for run in runs:
                row = [run.get("name", "")]
                row.extend(run.get("metrics", [{}])[0].get(k, "") for k in metric_keys)
                writer.writerow(row)

    def _metric_keys(self, runs: Sequence[Mapping[str, Any]]) -> List[str]:
        for run in runs:
            metrics = run.get("metrics") or []
            if metrics:
                return [k for k in metrics[0].keys() if k.lower() not in {"fold_id", "fold"}]
        return []

    def _make_radar_plot(self, runs: Sequence[Mapping[str, Any]], output_path: Path, style: str) -> None:
        metric_keys = self._metric_keys(runs)
        if not metric_keys:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("")
            return

        apply_paper_style(style, dpi=DEFAULT_DPI)

        num_metrics = len(metric_keys)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        for run in runs:
            metrics = run.get("metrics", [{}])[0] if run.get("metrics") else {}
            values = [float(metrics.get(k, 0)) for k in metric_keys]
            values += values[:1]
            ax.plot(angles, values, label=run.get("name", "run"))
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_keys)
        ax.set_yticklabels([])
        ax.set_title("Run Comparison", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _render_comparison_html(runs: Sequence[Mapping[str, Any]], leaderboard_path: Path, radar_path: Path) -> str:
        run_list_items = "".join(
            f"<li>{run.get('name','')} (protocol: {run.get('manifest',{}).get('protocol_hash','')})</li>"
            for run in runs
        )
        return f"""
<!DOCTYPE html>
<html><head><meta charset='UTF-8'><title>Run Comparison</title></head>
<body>
  <h1>Run Comparison Dashboard</h1>
  <h2>Leaderboard</h2>
  <p>See {leaderboard_path.name}</p>
  <h2>Radar</h2>
  <img src="{radar_path.name}" alt="run comparison radar" />
  <h2>Runs</h2>
  <ul>{run_list_items}</ul>
</body></html>
"""


__all__ = ["ReportingEngine", "ReportingMode", "ReportOutputs"]
