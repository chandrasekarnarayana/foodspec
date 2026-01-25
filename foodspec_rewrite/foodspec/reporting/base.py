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
from foodspec.reporting.modes import ReportMode, get_mode_config, validate_artifacts


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
        manifest = RunManifest.load(artifacts.manifest_path)

        # Load protocol snapshot
        protocol_snapshot = manifest.protocol_snapshot

        # Load optional artifact tables
        metrics = cls._load_csv(artifacts.metrics_path)
        predictions = cls._load_csv(artifacts.predictions_path)
        qc = cls._load_csv(artifacts.qc_path)

        # Load trust outputs if present
        trust_outputs = {}
        if artifacts.root.joinpath("trust_outputs.json").exists():
            trust_outputs = json.loads(artifacts.root.joinpath("trust_outputs.json").read_text())

        # Collect figures
        figures = collect_figures(run_dir)

        return cls(
            run_dir=run_dir,
            manifest=manifest,
            protocol_snapshot=protocol_snapshot,
            metrics=metrics,
            predictions=predictions,
            qc=qc,
            trust_outputs=trust_outputs,
            figures=figures,
        )

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

        # Render template
        template = self.env.get_template("base.html")
        html_content = template.render(context_dict)

        # Write to disk
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content)

        return out_path


def collect_figures(run_dir: Path) -> Dict[str, List[Path]]:
    """Index all figures under artifacts/viz folder.

    Recursively scans the viz directory and groups figures by subdirectory
    (e.g., drift, interpretability, uncertainty, pipeline).

    Parameters
    ----------
    run_dir : Path
        Path to the run directory.

    Returns
    -------
    dict of str to list of Path
        Mapping from category (subdirectory name) to list of image paths.
        Empty dict if viz directory doesn't exist.

    Examples
    --------
    Collect all figures from a run::

        figures = collect_figures(Path("/tmp/run"))
        print(figures["drift"])  # [Path(...), Path(...)]
    """
    figures: Dict[str, List[Path]] = {}
    viz_dir = run_dir / "plots" / "viz"

    if not viz_dir.exists():
        return figures

    # Recursively find all image files
    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".gif"}

    for category_dir in viz_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        image_paths = []

        for img_path in category_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions and img_path.is_file():
                image_paths.append(img_path)

        if image_paths:
            figures[category_name] = sorted(image_paths)

    return figures
