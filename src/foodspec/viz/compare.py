"""Multi-run comparison utilities for analysis tracking and benchmarking.

Compare multiple analysis runs to identify best models, track performance
over time, and monitor metric trends.

Usage:
    from foodspec.viz.compare import scan_runs, load_run_summary, compare_runs

    # Scan for runs
    runs = scan_runs("analysis_runs/")

    # Load summaries
    summaries = [load_run_summary(run) for run in runs]

    # Generate comparison dashboard
    compare_runs(summaries, output_dir="comparison_output")
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class RunSummary:
    """Summary of an analysis run.

    Attributes
    ----------
    run_id : str
        Unique run identifier
    run_dir : Path
        Path to run directory
    timestamp : str
        Run timestamp
    model_name : str
        Model name or algorithm
    validation_scheme : str
        Validation approach used
    metrics : dict
        Performance metrics
    trust_metrics : dict
        Trust/uncertainty metrics
    qc_flags : dict
        Quality control flags
    """

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        timestamp: str,
        model_name: str,
        validation_scheme: str,
        metrics: dict[str, float],
        trust_metrics: dict[str, float] | None = None,
        qc_flags: dict[str, bool] | None = None,
    ) -> None:
        """Initialize run summary."""
        self.run_id = run_id
        self.run_dir = run_dir
        self.timestamp = timestamp
        self.model_name = model_name
        self.validation_scheme = validation_scheme
        self.metrics = metrics or {}
        self.trust_metrics = trust_metrics or {}
        self.qc_flags = qc_flags or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "validation_scheme": self.validation_scheme,
            "metrics": self.metrics,
            "trust_metrics": self.trust_metrics,
            "qc_flags": self.qc_flags,
        }

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get metric value."""
        return self.metrics.get(name, default)

    def get_trust_metric(self, name: str, default: float = 0.0) -> float:
        """Get trust metric value."""
        return self.trust_metrics.get(name, default)


def scan_runs(root_dir: str | Path, pattern: str = "*/manifest.json") -> list[Path]:
    """Scan for run directories containing manifest.json.

    Parameters
    ----------
    root_dir : str | Path
        Root directory to scan
    pattern : str, default "*/manifest.json"
        Glob pattern to match run directories

    Returns
    -------
    list[Path]
        List of run directories (parent directories of manifest files)

    Examples
    --------
    >>> runs = scan_runs("analysis_runs/")
    >>> print(f"Found {len(runs)} runs")
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    # Find all manifest files
    manifest_files = list(root_dir.glob(pattern))

    # Get parent directories (run directories)
    run_dirs = [manifest_file.parent for manifest_file in manifest_files]

    return sorted(run_dirs)


def load_run_summary(run_dir: str | Path) -> RunSummary:
    """Load run summary from directory.

    Extracts key metrics, trust metrics, QC flags, model name,
    validation scheme, and timestamp from run artifacts.

    Parameters
    ----------
    run_dir : str | Path
        Path to run directory

    Returns
    -------
    RunSummary
        Run summary object

    Raises
    ------
    FileNotFoundError
        If manifest.json not found
    ValueError
        If manifest is malformed

    Examples
    --------
    >>> summary = load_run_summary("run_001")
    >>> print(f"Model: {summary.model_name}")
    >>> print(f"F1 Score: {summary.get_metric('macro_f1')}")
    """
    run_dir = Path(run_dir)

    # Load manifest
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Extract basic info
    run_id = manifest.get("run_id", run_dir.name)
    timestamp = manifest.get("timestamp", "")
    model_name = manifest.get("algorithm", "Unknown")
    validation_scheme = manifest.get("validation_scheme", "unknown")

    # Load metrics
    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Load trust metrics
    trust_metrics = {}
    trust_path = run_dir / "uncertainty_metrics.json"
    if trust_path.exists():
        with open(trust_path) as f:
            trust_metrics = json.load(f)

    # Load QC flags
    qc_flags = {}
    qc_path = run_dir / "qc_results.json"
    if qc_path.exists():
        with open(qc_path) as f:
            qc_data = json.load(f)
            # Extract boolean flags
            if isinstance(qc_data, dict):
                qc_flags = {k: bool(v) for k, v in qc_data.items() if isinstance(v, (bool, int, float))}

    return RunSummary(
        run_id=run_id,
        run_dir=run_dir,
        timestamp=timestamp,
        model_name=model_name,
        validation_scheme=validation_scheme,
        metrics=metrics,
        trust_metrics=trust_metrics,
        qc_flags=qc_flags,
    )


def create_leaderboard(
    summaries: Sequence[RunSummary],
    sort_by: tuple[str, ...] = ("macro_f1", "coverage"),
    ascending: tuple[bool, ...] = (False, False),
) -> pd.DataFrame:
    """Create leaderboard table from run summaries.

    Parameters
    ----------
    summaries : Sequence[RunSummary]
        Run summaries to compare
    sort_by : tuple[str, ...], default ("macro_f1", "coverage")
        Metrics to sort by (in order of priority)
    ascending : tuple[bool, ...], default (False, False)
        Sort direction for each metric

    Returns
    -------
    pd.DataFrame
        Leaderboard table sorted by specified metrics

    Examples
    --------
    >>> leaderboard = create_leaderboard(summaries)
    >>> print(leaderboard.head())
    """
    if not summaries:
        return pd.DataFrame()

    # Build rows
    rows = []
    for summary in summaries:
        row = {
            "run_id": summary.run_id,
            "model": summary.model_name,
            "validation": summary.validation_scheme,
            "timestamp": summary.timestamp,
        }

        # Add metrics
        for key, val in summary.metrics.items():
            row[key] = val

        # Add trust metrics
        for key, val in summary.trust_metrics.items():
            row[f"trust_{key}"] = val

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by specified columns
    sort_cols = [col for col in sort_by if col in df.columns]
    if sort_cols:
        df = df.sort_values(
            by=sort_cols,
            ascending=list(ascending[: len(sort_cols)]),
        )

    # Add rank column
    df.insert(0, "rank", range(1, len(df) + 1))

    return df.reset_index(drop=True)


def create_radar_plot(
    summaries: Sequence[RunSummary],
    metrics: Sequence[str] = ("macro_f1", "auroc", "coverage"),
    top_n: int = 5,
    output_path: str | Path | None = None,
) -> Path | None:
    """Create radar plot comparing top N runs.

    Parameters
    ----------
    summaries : Sequence[RunSummary]
        Run summaries to compare
    metrics : Sequence[str], default ("macro_f1", "auroc", "coverage")
        Metrics to plot on radar
    top_n : int, default 5
        Number of top runs to include
    output_path : str | Path, optional
        Output file path (default: radar.png)

    Returns
    -------
    Path or None
        Path to saved plot, or None if not saved

    Examples
    --------
    >>> create_radar_plot(summaries, top_n=3, output_path="radar.png")
    """
    if not summaries:
        return None

    # Sort by macro_f1
    sorted_summaries = sorted(
        summaries,
        key=lambda s: s.get_metric("macro_f1", 0),
        reverse=True,
    )[:top_n]

    # Prepare data
    labels = list(metrics)
    num_vars = len(labels)

    # Compute angles for radar
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    # Plot each run
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    for i, summary in enumerate(sorted_summaries):
        values = []
        for metric in metrics:
            val = summary.get_metric(metric, 0)
            # Transform some metrics (e.g., 1-ECE, 1-abstain_rate)
            if metric.startswith("1-"):
                actual_metric = metric[2:]
                val = 1.0 - summary.get_metric(actual_metric, 0)
            values.append(val)

        values += values[:1]  # Close the plot

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=f"{summary.run_id} ({summary.model_name})",
            color=colors[i],
        )
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Run Comparison", size=16, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    return None


def compute_baseline_deltas(
    summaries: Sequence[RunSummary],
    baseline_id: str,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute deltas from baseline run.

    Parameters
    ----------
    summaries : Sequence[RunSummary]
        Run summaries to compare
    baseline_id : str
        Run ID to use as baseline
    metrics : Sequence[str], optional
        Metrics to compute deltas for (default: all numeric metrics)

    Returns
    -------
    pd.DataFrame
        DataFrame with delta columns for each metric

    Examples
    --------
    >>> deltas = compute_baseline_deltas(summaries, baseline_id="run_001")
    >>> print(deltas[["run_id", "delta_macro_f1", "delta_auroc"]])
    """
    if not summaries:
        return pd.DataFrame()

    # Find baseline
    baseline = None
    for summary in summaries:
        if summary.run_id == baseline_id:
            baseline = summary
            break

    if baseline is None:
        raise ValueError(f"Baseline run not found: {baseline_id}")

    # Determine metrics
    if metrics is None:
        metrics = list(baseline.metrics.keys())

    # Build delta table
    rows = []
    for summary in summaries:
        row = {
            "run_id": summary.run_id,
            "model": summary.model_name,
            "is_baseline": summary.run_id == baseline_id,
        }

        for metric in metrics:
            baseline_val = baseline.get_metric(metric, 0)
            current_val = summary.get_metric(metric, 0)
            delta = current_val - baseline_val
            row[f"delta_{metric}"] = delta
            row[metric] = current_val

        rows.append(row)

    return pd.DataFrame(rows)


def create_monitoring_plot(
    summaries: Sequence[RunSummary],
    metrics: Sequence[str] = ("macro_f1", "auroc"),
    output_path: str | Path | None = None,
) -> Path | None:
    """Create monitoring plot showing metric trends over time.

    Parameters
    ----------
    summaries : Sequence[RunSummary]
        Run summaries to plot
    metrics : Sequence[str], default ("macro_f1", "auroc")
        Metrics to plot
    output_path : str | Path, optional
        Output file path

    Returns
    -------
    Path or None
        Path to saved plot, or None if not saved

    Examples
    --------
    >>> create_monitoring_plot(summaries, output_path="monitoring.png")
    """
    if not summaries:
        return None

    # Sort by timestamp
    sorted_summaries = sorted(summaries, key=lambda s: s.timestamp)

    # Parse timestamps
    timestamps = []
    for summary in sorted_summaries:
        try:
            ts = datetime.fromisoformat(summary.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.now()
        timestamps.append(ts)

    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [s.get_metric(metric, 0) for s in sorted_summaries]

        ax.plot(timestamps, values, "o-", linewidth=2, markersize=8)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Add model labels
        for i, summary in enumerate(sorted_summaries):
            ax.annotate(
                summary.model_name,
                (timestamps[i], values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                alpha=0.7,
            )

    axes[-1].set_xlabel("Timestamp", fontsize=12)
    fig.suptitle("Metric Monitoring Over Time", fontsize=16, weight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    return None


def create_comparison_dashboard(
    summaries: Sequence[RunSummary],
    output_path: str | Path,
    baseline_id: str | None = None,
) -> Path:
    """Create HTML comparison dashboard.

    Parameters
    ----------
    summaries : Sequence[RunSummary]
        Run summaries to compare
    output_path : str | Path
        Output HTML file path
    baseline_id : str, optional
        Baseline run ID for delta computation

    Returns
    -------
    Path
        Path to created HTML file

    Examples
    --------
    >>> create_comparison_dashboard(
    ...     summaries,
    ...     "comparison_dashboard.html",
    ...     baseline_id="run_001"
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create leaderboard
    leaderboard = create_leaderboard(summaries)

    # Build HTML (using double braces for CSS to avoid format issues)
    html_template = """<!DOCTYPE html>
<html>
<head>
    <title>FoodSpec Run Comparison</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; margin-top: 30px; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metric {{ font-weight: bold; }}
        .summary {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }}
    </style>
</head>
<body>
    <h1>🥗 FoodSpec Multi-Run Comparison</h1>

    <div class="summary">
        <strong>Total Runs:</strong> {total_runs}<br>
        <strong>Date Generated:</strong> {date}
    </div>

    <h2>Leaderboard</h2>
    {leaderboard_html}
</body>
</html>
"""

    # Format leaderboard as HTML
    leaderboard_html = leaderboard.to_html(index=False, float_format="%.4f")

    # Fill template
    html = html_template.format(
        total_runs=len(summaries),
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        leaderboard_html=leaderboard_html,
    )

    # Write file
    with open(output_path, "w") as f:
        f.write(html)

    return output_path.resolve()


def compare_runs(
    summaries: Sequence[RunSummary],
    output_dir: str | Path,
    baseline_id: str | None = None,
    top_n: int = 5,
) -> dict[str, Path]:
    """Compare runs and generate all outputs.

    Generates:
    - comparison_dashboard.html: Interactive dashboard
    - comparison.csv: Leaderboard table
    - radar.png: Radar plot of top N runs
    - monitoring.png: Metric trends over time
    - baseline_deltas.csv: Deltas from baseline (if specified)

    Parameters
    ----------
    summaries : Sequence[RunSummary]
        Run summaries to compare
    output_dir : str | Path
        Output directory for results
    baseline_id : str, optional
        Baseline run ID for delta computation
    top_n : int, default 5
        Number of top runs for radar plot

    Returns
    -------
    dict[str, Path]
        Dictionary mapping output type to file path

    Examples
    --------
    >>> outputs = compare_runs(
    ...     summaries,
    ...     output_dir="comparison_results",
    ...     baseline_id="run_001"
    ... )
    >>> print(f"Dashboard: {outputs['dashboard']}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Create leaderboard CSV
    leaderboard = create_leaderboard(summaries)
    csv_path = output_dir / "comparison.csv"
    leaderboard.to_csv(csv_path, index=False)
    outputs["leaderboard"] = csv_path

    # Create dashboard HTML
    dashboard_path = output_dir / "comparison_dashboard.html"
    create_comparison_dashboard(summaries, dashboard_path, baseline_id)
    outputs["dashboard"] = dashboard_path

    # Create radar plot
    radar_path = output_dir / "radar.png"
    create_radar_plot(summaries, top_n=top_n, output_path=radar_path)
    outputs["radar"] = radar_path

    # Create monitoring plot
    monitoring_path = output_dir / "monitoring.png"
    create_monitoring_plot(summaries, output_path=monitoring_path)
    outputs["monitoring"] = monitoring_path

    # Create baseline deltas if specified
    if baseline_id:
        try:
            deltas = compute_baseline_deltas(summaries, baseline_id)
            deltas_path = output_dir / "baseline_deltas.csv"
            deltas.to_csv(deltas_path, index=False)
            outputs["baseline_deltas"] = deltas_path
        except ValueError as e:
            warnings.warn(
                f"Could not compute baseline deltas: {e}",
                UserWarning,
                stacklevel=2,
            )

    return outputs
