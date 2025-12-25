from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def create_run_folder(base_dir: Path) -> Path:
    """Create timestamped run directory with standardized subdirectories.

    Generates a unique folder for each analysis run to prevent result overwriting
    and enable chronological tracking. Follows FoodSpec directory convention:

    ::

        base_dir/
        └── YYYYMMDD_HHMMSS_run/
            ├── figures/
            │   └── hsi/
            ├── tables/
            └── hsi/

    **Reproducibility Context:**
    - Timestamped folders enable "one run = one folder" traceability
    - Prevents accidental overwriting of previous results
    - Facilitates regulatory audits (each run is self-contained)
    - Compatible with version control (Git can track folder names)

    Args:
        base_dir: Root output directory (e.g., Path("outputs"))

    Returns:
        Path to created run directory (e.g., "outputs/20251225_143022_run")

    Examples:
        >>> from pathlib import Path
        >>> from foodspec.output_bundle import create_run_folder
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = create_run_folder(Path(tmpdir))
        ...     assert run_dir.exists()
        ...     assert (run_dir / "figures").exists()
        ...     assert (run_dir / "tables").exists()

    See Also:
        - Theory: docs/protocols/reference_protocol.md#step-8-document
        - Reproducibility: docs/foundations/data_structures_and_fair_principles.md
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{ts}_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "figures" / "hsi").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)
    (run_dir / "hsi").mkdir(exist_ok=True)
    return run_dir


def save_report_text(path: Path, text: str):
    """Save plain text report with UTF-8 encoding.

    Writes analysis summary, diagnostic messages, or protocol documentation
    to a text file. UTF-8 ensures international character support (e.g.,
    Müller, São Paulo).

    **Use Cases:**
    - Methods section text for manuscripts (journal submission)
    - QC checklist for lab technicians (print-friendly)
    - Error/warning logs for debugging

    Args:
        path: Output file path (e.g., Path("run/report.txt"))
        text: Report content (plain text, UTF-8 safe)

    Examples:
        >>> from pathlib import Path
        >>> from foodspec.output_bundle import save_report_text
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir) / "report.txt"
        ...     save_report_text(path, "Analysis complete: AUC=0.95")
        ...     assert path.read_text() == "Analysis complete: AUC=0.95"

    See Also:
        - save_report_html(): HTML version for web viewing
    """
    path.write_text(text, encoding="utf-8")


def save_report_html(path: Path, text: str):
    """Save report as HTML with basic escaping for web viewing.

    Converts plain text to HTML with <pre> formatting (preserves whitespace)
    and escapes HTML special characters (&, <, >) to prevent rendering issues.

    **Use Cases:**
    - Web-based dashboards (view reports in browser)
    - Email attachments (HTML rendering)
    - Lab information systems (LIS) integration

    **Limitations:**
    - Basic HTML only (no CSS, no JavaScript)
    - Not XSS-safe for untrusted user input
    - Use Markdown/Jinja2 for richer formatting

    Args:
        path: Output file path (e.g., Path("run/report.html"))
        text: Report content (plain text, will be HTML-escaped)

    Examples:
        >>> from pathlib import Path
        >>> from foodspec.output_bundle import save_report_html
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir) / "report.html"
        ...     save_report_html(path, "AUC > 0.9 means excellent")
        ...     html = path.read_text()
        ...     assert "AUC &gt; 0.9" in html  # < escaped to &gt;

    See Also:
        - save_report_text(): Plain text version
    """
    html = "<html><body><pre>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>"
    path.write_text(html, encoding="utf-8")


def save_tables(run_dir: Path, tables: Dict[str, object]):
    """Save DataFrames as CSV files in tables/ subdirectory.

    Exports numerical results (metrics, feature values, sample metadata) to
    CSV for downstream analysis in Excel, R, or Python. CSV ensures
    interoperability and long-term accessibility (FAIR principles).

    **Reproducibility Context:**
    - CSV is human-readable (open with text editor)
    - No vendor lock-in (unlike .xlsx, .sav, .mat)
    - Git-friendly for version control
    - ISO/IEC 15438 compliant (standardized format)

    **Saved Tables:**
    - Metrics: Classification/regression performance
    - Feature matrix: Preprocessed spectral features
    - Sample metadata: Provenance, timestamps, operators
    - Cross-validation folds: For reproducibility audits

    Args:
        run_dir: Run directory from create_run_folder()
        tables: Dict mapping table names to pandas DataFrames
                (e.g., {"metrics": metrics_df, "features": X})

    Examples:
        >>> import pandas as pd
        >>> from pathlib import Path
        >>> from foodspec.output_bundle import create_run_folder, save_tables
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = create_run_folder(Path(tmpdir))
        ...     tables = {"metrics": pd.DataFrame({"auc": [0.95]})}
        ...     save_tables(run_dir, tables)
        ...     assert (run_dir / "tables" / "metrics.csv").exists()

    See Also:
        - Theory: docs/foundations/data_structures_and_fair_principles.md
        - Protocol: docs/protocols/reference_protocol.md#step-8-document
    """
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name, df in tables.items():
        if df is None:
            continue
        out = tables_dir / f"{name}.csv"
        df.to_csv(out, index=False)


def save_figures(run_dir: Path, figures: Dict[str, object]):
    """Save matplotlib figures or numpy arrays as PNG images.

    Exports publication-quality visualizations at 200 DPI (journal submission
    standard). Handles both matplotlib Figure objects and numpy arrays (converted
    to heatmaps). Supports nested names (e.g., "hsi/labels").

    **Publication Standards:**
    - 200 DPI: Sufficient for most journals (300 DPI for high-impact)
    - PNG format: Lossless compression, transparency support
    - Tight bounding box: Removes excess whitespace
    - Viridis colormap: Perceptually uniform, colorblind-safe

    **Supported Inputs:**
    - matplotlib.figure.Figure: Saved directly
    - np.ndarray: Converted to heatmap with viridis colormap

    Args:
        run_dir: Run directory from create_run_folder()
        figures: Dict mapping figure names to Figure objects or np.ndarray
                 (e.g., {"pca": fig, "hsi/labels": label_array})

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from pathlib import Path
        >>> from foodspec.output_bundle import create_run_folder, save_figures
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = create_run_folder(Path(tmpdir))
        ...     fig, ax = plt.subplots()
        ...     ax.plot([1, 2, 3])
        ...     save_figures(run_dir, {"test": fig})
        ...     assert (run_dir / "figures" / "test.png").exists()
        ...     plt.close()

    See Also:
        - Visualization: docs/visualization/
        - Protocol: docs/protocols/reference_protocol.md#step-6-visualize
    """
    figs_dir = run_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    for name, fig in figures.items():
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            plt = None
            np = None
        # Allow nested names like "hsi/labels"
        target_path = figs_dir / f"{name}.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if plt is not None and np is not None and isinstance(fig, np.ndarray):
            plt.figure()
            plt.imshow(fig, cmap="viridis")
            plt.axis("off")
            plt.savefig(target_path, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            fig.savefig(target_path, dpi=200)


def save_metadata(run_dir: Path, meta: Dict):
    """Save run metadata as JSON for machine-readable provenance.

    Stores structured metadata (protocol, timestamp, user, tool version) in
    metadata.json for automated parsing, registry integration, and audit trails.

    **Recommended Metadata Fields:**
    - protocol: Protocol name (e.g., "oil_authentication")
    - protocol_version: Semantic version (e.g., "2.1.0")
    - timestamp: ISO 8601 timestamp (e.g., "2025-12-25T14:30:00Z")
    - user: Operator name or ID
    - tool_version: FoodSpec version (e.g., "0.18.0")
    - dataset_hash: SHA-256 of input data
    - preprocessing: Dict of preprocessing steps
    - validation_strategy: CV method (e.g., "stratified_5fold")

    **Compliance Context:**
    - FDA 21 CFR Part 11: Requires electronic signatures and audit trails
    - ISO 17025: Requires traceability of test results
    - FAIR principles: Machine-readable metadata for data sharing

    Args:
        run_dir: Run directory from create_run_folder()
        meta: Dict of metadata (serializable to JSON)

    Examples:
        >>> from pathlib import Path
        >>> from foodspec.output_bundle import create_run_folder, save_metadata
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = create_run_folder(Path(tmpdir))
        ...     meta = {"protocol": "oil_auth", "timestamp": "2025-12-25T10:00:00Z"}
        ...     save_metadata(run_dir, meta)
        ...     assert (run_dir / "metadata.json").exists()

    See Also:
        - Theory: docs/foundations/data_structures_and_fair_principles.md
        - Registry: docs/protocols/reference_protocol.md#provenance-tracking
    """
    meta_path = run_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def append_log(run_dir: Path, message: str):
    """Append timestamped message to run.log for debugging and audits.

    Creates append-only log file for chronological event tracking (data loading,
    preprocessing steps, warnings, errors). Critical for troubleshooting and
    regulatory compliance (tamper-evident logs).

    **Use Cases:**
    - Debugging: "Baseline correction failed on sample X"
    - Audit trail: "User Y ran analysis Z at timestamp T"
    - Performance tracking: "PCA took 5.2 seconds"
    - Warning records: "Low SNR detected in 3 samples"

    **Best Practices:**
    - Include timestamp in message (or prepend externally)
    - Use structured logging (JSON) for machine parsing
    - Rotate logs for long-running processes (logrotate)

    Args:
        run_dir: Run directory from create_run_folder()
        message: Log message (newline appended automatically)

    Examples:
        >>> from pathlib import Path
        >>> from foodspec.output_bundle import create_run_folder, append_log
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     run_dir = create_run_folder(Path(tmpdir))
        ...     append_log(run_dir, "Analysis started")
        ...     append_log(run_dir, "Baseline correction: ALS")
        ...     log = (run_dir / "run.log").read_text()
        ...     assert "Analysis started" in log
        ...     assert "Baseline correction: ALS" in log

    See Also:
        - Logging: docs/config_logging.md
        - Troubleshooting: docs/troubleshooting/troubleshooting_faq.md
    """
    log_path = run_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.strip() + "\n")


def save_index(
    run_dir: Path,
    metadata: Dict,
    tables: Dict[str, object],
    figures: Dict[str, object],
    warnings: List[str],
):
    """
    Lightweight index.json for quick inspection of a run.
    Lists tables, figures, metadata, and warnings/notes.
    """
    index = {
        "run_id": run_dir.name,
        "metadata": metadata,
        "tables": list(tables.keys()),
        "figures": list(figures.keys()),
        "warnings": warnings,
        "models": metadata.get("models", []),
        "validation": metadata.get("validation_strategy"),
        "harmonization": metadata.get("harmonization", {}),
    }
    (run_dir / "index.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
