"""
FoodSpec Report Context and Builder.

ReportContext: Loads artifacts from a run directory (manifest, protocol, metrics, etc.)
ReportBuilder: Builds HTML reports with sidebar navigation, artifact validation, and image embedding.

Examples
--------
Load a run and build an HTML report::

    context = ReportContext.load(Path("/tmp/run"))
    builder = ReportBuilder(context)
    html_path = builder.build_html(Path("/tmp/run/report.html"), mode=ReportMode.RESEARCH)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from jinja2 import Environment, PackageLoader, select_autoescape

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.core.manifest import RunManifest
from foodspec.report.sections.multivariate import build_multivariate_section
from foodspec.reporting.modes import ReportMode, get_mode_config, validate_artifacts

# Jinja2 template loader
_JINJA_ENV = Environment(
    loader=PackageLoader("foodspec.reporting", "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


class ReportContext:
    """Loads and provides access to run artifacts.

    Loads manifest, protocol snapshot, and standard artifact tables
    (metrics.csv, predictions.csv, qc.csv, trust outputs) from a run directory.

    Examples
    --------
    Load context and inspect artifacts::

        context = ReportContext.load(Path("/tmp/run"))
        print(context.manifest.seed)
        print(context.metrics[:5])
        print(context.available_artifacts)
    """

    def __init__(
        self,
        run_dir: Path,
        manifest: RunManifest,
        protocol_snapshot: Mapping[str, Any],
        metrics: Optional[List[Dict[str, Any]]] = None,
        predictions: Optional[List[Dict[str, Any]]] = None,
        qc: Optional[List[Dict[str, Any]]] = None,
        trust_outputs: Optional[Dict[str, Any]] = None,
        figures: Optional[Dict[str, List[Path]]] = None,
        multivariate: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        """Initialize report context.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.
        manifest : RunManifest
            Execution manifest with metadata.
        protocol_snapshot : Mapping[str, Any]
            Expanded protocol configuration.
        metrics : list of dict, optional
            Parsed metrics.csv rows.
        predictions : list of dict, optional
            Parsed predictions.csv rows.
        qc : list of dict, optional
            Parsed qc.csv rows.
        trust_outputs : dict, optional
            Parsed trust outputs (uncertainty, abstention, etc.).
        figures : dict of list of Path, optional
            Indexed figures by category (e.g., {"drift": [path1, path2]}).
        """
        self.run_dir = run_dir
        self.manifest = manifest
        self.protocol_snapshot = protocol_snapshot
        self.metrics = metrics or []
        self.predictions = predictions or []
        self.qc = qc or []
        self.trust_outputs = trust_outputs or {}
        self.figures = figures or {}
        self.multivariate = multivariate or {}

    @classmethod
    def load(cls, run_dir: Path) -> ReportContext:
        """Load context from a run directory.

        Loads manifest, protocol snapshot, and available artifact tables.
        Missing tables are loaded as empty lists.

        Parameters
        ----------
        run_dir : Path
            Path to the run directory.

        Returns
        -------
        ReportContext
            Loaded context.

        Raises
        ------
        FileNotFoundError
            If manifest.json is missing.
        """
        artifacts = ArtifactRegistry(run_dir)

        # Load manifest
        if not artifacts.manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {run_dir}")
        manifest = cls._load_manifest(artifacts.manifest_path)

        # Load protocol snapshot
        protocol_snapshot = manifest.protocol_snapshot

        # Load optional artifact tables
        metrics = cls._load_csv(artifacts.metrics_path)
        if not metrics:
            metrics = cls._load_csv(run_dir / "tables" / "metrics.csv")
        predictions = cls._load_csv(artifacts.predictions_path)
        if not predictions:
            predictions = cls._load_csv(run_dir / "tables" / "predictions.csv")
        qc = cls._load_csv(artifacts.qc_path)
        if not qc:
            qc = cls._load_csv(run_dir / "tables" / "qc.csv")

        # Load trust outputs if present (multiple possible locations)
        trust_outputs = cls._load_trust_outputs(run_dir)

        # Collect figures
        figures = collect_figures(run_dir)
        multivariate = cls._load_multivariate_tables(run_dir)

        return cls(
            run_dir=run_dir,
            manifest=manifest,
            protocol_snapshot=protocol_snapshot,
            metrics=metrics,
            predictions=predictions,
            qc=qc,
            trust_outputs=trust_outputs,
            figures=figures,
            multivariate=multivariate,
        )

    @staticmethod
    def _load_trust_outputs(run_dir: Path) -> Dict[str, Any]:
        """Load trust outputs from various locations in run directory.

        Scans for:
        - trust_outputs.json (legacy)
        - trust/calibration.json
        - trust/conformal.json
        - trust/abstention.json
        - trust/coverage.json
        - trust/reliability.json
        - drift/batch_drift.json
        - drift/temporal_drift.json
        - drift/stage_differences.json
        - qc/qc_summary.json

        Returns consolidated dict with all available trust/drift/qc outputs.
        """
        trust_data: Dict[str, Any] = {}

        # Legacy location
        legacy_path = run_dir / "trust_outputs.json"
        if legacy_path.exists():
            try:
                trust_data = json.loads(legacy_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        # Trust subdirectory
        trust_dir = run_dir / "trust"
        if trust_dir.exists():
            for trust_file in ["calibration.json", "conformal.json", "abstention.json",
                              "coverage.json", "reliability.json", "readiness.json"]:
                file_path = trust_dir / trust_file
                if file_path.exists():
                    try:
                        key = trust_file.replace(".json", "")
                        trust_data[key] = json.loads(file_path.read_text())
                    except (json.JSONDecodeError, OSError):
                        pass

        # Drift subdirectory
        drift_dir = run_dir / "drift"
        if drift_dir.exists():
            drift_data = {}
            for drift_file in ["batch_drift.json", "temporal_drift.json",
                             "stage_differences.json", "replicate_similarity.json"]:
                file_path = drift_dir / drift_file
                if file_path.exists():
                    try:
                        key = drift_file.replace(".json", "")
                        drift_data[key] = json.loads(file_path.read_text())
                    except (json.JSONDecodeError, OSError):
                        pass
            if drift_data:
                trust_data["drift"] = drift_data

        # QC subdirectory
        qc_dir = run_dir / "qc"
        if qc_dir.exists():
            qc_summary_path = qc_dir / "qc_summary.json"
            if qc_summary_path.exists():
                try:
                    trust_data["qc_summary"] = json.loads(qc_summary_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass
            qc_control_path = qc_dir / "control_charts.json"
            if qc_control_path.exists():
                try:
                    trust_data["qc_control_charts"] = json.loads(qc_control_path.read_text())
                except (json.JSONDecodeError, OSError):
                    pass

        return trust_data

    @staticmethod
    def _load_csv(path: Path) -> List[Dict[str, Any]]:
        """Load CSV file as list of dicts; return empty list if missing."""
        if not path.exists():
            return []
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows

    @staticmethod
    def _load_manifest(path: Path) -> RunManifest:
        """Load manifest.json with compatibility for multiple schemas."""
        try:
            return RunManifest.load(path)
        except Exception:
            raw = json.loads(path.read_text(encoding="utf-8"))
            protocol_snapshot = raw.get("protocol_snapshot", {})
            artifacts = raw.get("artifacts", {})
            seed = raw.get("seed")
            manifest = RunManifest.build(
                protocol_snapshot=protocol_snapshot,
                data_path=None,
                seed=seed,
                artifacts=artifacts,
            )
            manifest.protocol_hash = raw.get("protocol_hash") or raw.get("protocol_sha256") or manifest.protocol_hash
            manifest.data_fingerprint = raw.get("data_fingerprint", manifest.data_fingerprint)
            manifest.python_version = raw.get("python_version", manifest.python_version)
            manifest.platform = raw.get("platform", manifest.platform)
            manifest.start_time = raw.get("start_time") or raw.get("timestamp") or manifest.start_time
            manifest.end_time = raw.get("end_time", manifest.end_time)
            duration = raw.get("duration_seconds")
            if duration is not None:
                try:
                    manifest.duration_seconds = float(duration)
                except (TypeError, ValueError):
                    pass
            dependencies = dict(manifest.dependencies or {})
            if isinstance(raw.get("dependencies"), dict):
                dependencies.update(raw.get("dependencies", {}))
            if raw.get("foodspec_version"):
                dependencies.setdefault("foodspec", raw.get("foodspec_version"))
            manifest.dependencies = dependencies
            return manifest

    @classmethod
    def _load_multivariate_tables(cls, run_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Load optional multivariate tables emitted by the protocol."""
        names = [
            "multivariate_scores",
            "multivariate_loadings",
            "multivariate_summary",
            "multivariate_qc",
            "multivariate_group_shift",
        ]
        tables: Dict[str, List[Dict[str, Any]]] = {}
        for name in names:
            path = run_dir / "tables" / f"{name}.csv"
            rows = cls._load_csv(path)
            if rows:
                tables[name] = rows
        return tables

    @property
    def available_artifacts(self) -> List[str]:
        """Return list of available artifact types."""
        artifacts = [
            "manifest",  # manifest is always present
            "protocol_snapshot",  # always in manifest
            "data_fingerprint",  # always in manifest
        ]
        if self.metrics:
            artifacts.append("metrics")
        if self.predictions:
            artifacts.append("predictions")
        if self.qc:
            artifacts.append("qc")
        if self.trust_outputs:
            artifacts.append("trust_outputs")
        if self.figures:
            artifacts.extend(self.figures.keys())
        if self.multivariate:
            artifacts.append("multivariate")
        return artifacts

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dict for template rendering.

        Returns
        -------
        dict
            Serializable context dict with all data and metadata.
        """
        return {
            "manifest": {
                "seed": self.manifest.seed,
                "protocol_hash": self.manifest.protocol_hash,
                "data_fingerprint": self.manifest.data_fingerprint,
                "start_time": self.manifest.start_time,
                "end_time": self.manifest.end_time,
                "duration_seconds": self.manifest.duration_seconds,
                "python_version": self.manifest.python_version,
                "platform": self.manifest.platform,
            },
            "protocol": self.protocol_snapshot,
            "metrics": self.metrics,
            "predictions": self.predictions,
            "qc": self.qc,
            "trust_outputs": self.trust_outputs,
            "figures": self.figures,
            "multivariate": self.multivariate,
            "available_artifacts": self.available_artifacts,
        }


class ReportBuilder:
    """Builds HTML reports from ReportContext.

    Validates required artifacts for a mode, constructs an HTML report
    with sidebar navigation, and embeds images via relative paths.

    Examples
    --------
    Build an HTML report in regulatory mode::

        context = ReportContext.load(Path("/tmp/run"))
        builder = ReportBuilder(context)
        html_path = builder.build_html(
            Path("/tmp/run/report.html"),
            mode=ReportMode.REGULATORY
        )
    """

    def __init__(self, context: ReportContext) -> None:
        """Initialize builder.

        Parameters
        ----------
        context : ReportContext
            Loaded run context.
        """
        self.context = context
        self._setup_jinja2()

    def _setup_jinja2(self) -> None:
        """Initialize Jinja2 environment."""
        try:
            self.env = Environment(
                loader=PackageLoader("foodspec.reporting", "templates"),
                autoescape=select_autoescape(["html", "xml"]),
            )
        except (ValueError, ImportError):
            # Fallback: use filesystem loader
            template_dir = Path(__file__).parent / "templates"
            from jinja2 import FileSystemLoader
            self.env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )

    def build_html(
        self,
        out_path: Path,
        mode: ReportMode | str = ReportMode.RESEARCH,
        title: str = "FoodSpec Report",
    ) -> Path:
        """Build HTML report and write to disk.

        Validates required artifacts for the mode, then renders
        HTML with sidebar navigation and embedded figures.

        Parameters
        ----------
        out_path : Path
            Path where report HTML will be written.
        mode : ReportMode or str, optional
            Reporting mode (RESEARCH, REGULATORY, MONITORING).
            Default: RESEARCH.
        title : str, optional
            Report title. Default: "FoodSpec Report".

        Returns
        -------
        Path
            Path to written report.

        Raises
        ------
        ValueError
            If required artifacts are missing for the mode.
        """
        # Convert string mode to enum
        if isinstance(mode, str):
            mode = ReportMode(mode)

        # Get mode config
        mode_config = get_mode_config(mode)

        # Validate required artifacts
        available = self.context.available_artifacts
        is_valid, missing = validate_artifacts(
            mode=mode,
            available_artifacts=available,
            warnings_as_errors=mode_config.warnings_as_errors,
        )

        if not is_valid and mode_config.warnings_as_errors:
            raise ValueError(
                f"Mode {mode.value} requires artifacts: {mode_config.required_artifacts}. "
                f"Missing: {missing}. "
                f"Available: {available}"
            )

        # Prepare template data
        context_dict = self.context.to_dict()
        context_dict.update({
            "title": title,
            "mode": mode.value,
            "mode_description": mode_config.description,
            "enabled_sections": mode_config.enabled_sections,
        })

        mv_section = build_multivariate_section(self.context)
        context_dict["multivariate_section"] = mv_section
        context_dict["figures"] = _prepare_figures(self.context.figures, out_path.parent)

        from foodspec.reporting.cards import build_experiment_card

        card = build_experiment_card(self.context, mode=mode)
        context_dict["card"] = {
            "run_id": card.run_id,
            "timestamp": card.timestamp,
            "task": card.task,
            "modality": card.modality,
            "model": card.model,
            "validation_scheme": card.validation_scheme,
            "macro_f1": card.macro_f1,
            "auroc": card.auroc,
            "ece": card.ece,
            "coverage": card.coverage,
            "abstain_rate": card.abstain_rate,
            "mean_set_size": card.mean_set_size,
            "drift_score": card.drift_score,
            "qc_pass_rate": card.qc_pass_rate,
            "confidence_level": card.confidence_level.value,
            "confidence_reasoning": card.confidence_reasoning,
            "deployment_readiness": card.deployment_readiness.value,
            "readiness_reasoning": card.readiness_reasoning,
            "key_risks": card.key_risks,
            "auto_summary": card.auto_summary,
        }

        # Render template
        template = self.env.get_template("base.html")
        html_content = template.render(context_dict)

        # Write to disk
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content)

        return out_path


def collect_figures(run_dir: Path) -> Dict[str, List[Path]]:
    """Index all figures under multiple artifact folders.

    Scans multiple directories for figures:
    - plots/viz/*
    - figures/*
    - trust/plots/*
    - drift/plots/*
    - qc/plots/*

    Recursively scans each directory and groups figures by subdirectory
    (e.g., drift, interpretability, uncertainty, pipeline).

    Parameters
    ----------
    run_dir : Path
        Path to the run directory.

    Returns
    -------
    dict of str to list of Path
        Mapping from category (subdirectory name) to list of image paths.
        Empty dict if no figures found.

    Examples
    --------
    Collect all figures from a run::

        figures = collect_figures(Path("/tmp/run"))
        print(figures["drift"])  # [Path(...), Path(...)]
    """
    figures: Dict[str, List[Path]] = {}
    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".gif"}

    # Directories to scan for figures
    scan_dirs = [
        run_dir / "plots" / "viz",
        run_dir / "figures",
        run_dir / "trust" / "plots",
        run_dir / "drift" / "plots",
        run_dir / "qc" / "plots",
        run_dir / "plots",  # Catch-all for any plots/ directory
    ]

    for base_dir in scan_dirs:
        if not base_dir.exists():
            continue

        # If base_dir is the figures/ or plots/ root, scan directly
        if base_dir.name in {"figures", "plots"}:
            for img_path in base_dir.rglob("*"):
                if img_path.suffix.lower() in image_extensions and img_path.is_file():
                    # Get category from parent directory or use "general"
                    if img_path.parent == base_dir:
                        category = "general"
                    else:
                        category = img_path.parent.name
                    figures.setdefault(category, []).append(img_path)
        else:
            # For categorized dirs (viz, trust/plots, etc), use parent as category
            for img_path in base_dir.rglob("*"):
                if img_path.suffix.lower() in image_extensions and img_path.is_file():
                    # Category from direct parent of base_dir
                    category = base_dir.parent.name if base_dir.parent != run_dir else base_dir.name
                    figures.setdefault(category, []).append(img_path)

    # Sort figure lists
    for category in figures:
        figures[category] = sorted(set(figures[category]))  # Remove duplicates

    return figures


def _prepare_figures(
    figures: Dict[str, List[Path]],
    base_dir: Path,
) -> Dict[str, List[Dict[str, str]]]:
    payload: Dict[str, List[Dict[str, str]]] = {}
    for category, paths in figures.items():
        entries: List[Dict[str, str]] = []
        for path in paths:
            try:
                rel = path.relative_to(base_dir)
            except ValueError:
                rel = path
            entries.append(
                {
                    "path": str(rel).replace("\\", "/"),
                    "name": path.stem,
                }
            )
        if entries:
            payload[category] = entries
    return payload
