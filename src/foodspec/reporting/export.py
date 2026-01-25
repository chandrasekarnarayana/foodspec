"""Reproducibility and archive export system.

Build reproducibility packs and create stable, shareable archives
of analysis runs for publication and distribution.

Usage:
    from foodspec.reporting.export import build_reproducibility_pack, export_archive
    
    # Build reproducibility pack
    pack_dir = build_reproducibility_pack(
        run_dir="path/to/run",
        out_dir="path/to/pack"
    )
    
    # Export stable archive
    archive_path = export_archive(
        out_zip_path="analysis_run.zip",
        run_dir="path/to/run",
        include=("dossier", "figures", "tables", "bundle")
    )
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence


class ReproducibilityPackBuilder:
    """Build reproducibility packs for analysis runs.
    
    A reproducibility pack contains:
    - Protocol snapshot (expanded YAML/JSON)
    - Manifest with execution metadata
    - Environment freeze (pip dependencies)
    - Metrics, predictions, and QC tables
    - Index of all plots and outputs
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self.run_dir: Path | None = None
        self.out_dir: Path | None = None

    def build(self, run_dir: str | Path, out_dir: str | Path) -> Path:
        """Build reproducibility pack.
        
        Parameters
        ----------
        run_dir : str | Path
            Root directory of analysis run
        out_dir : str | Path
            Output directory for reproducibility pack
            
        Returns
        -------
        Path
            Path to reproducibility pack directory
            
        Raises
        ------
        FileNotFoundError
            If run_dir doesn't exist
        """
        self.run_dir = Path(run_dir)
        self.out_dir = Path(out_dir)

        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Generate pack components
        self._copy_manifest()
        self._export_protocol_snapshot()
        self._create_environment_freeze()
        self._copy_data_tables()
        self._create_plots_index()
        self._create_pack_metadata()

        return self.out_dir

    def _copy_manifest(self) -> None:
        """Copy manifest.json to pack."""
        manifest_src = self.run_dir / "manifest.json"
        if manifest_src.exists():
            manifest_dst = self.out_dir / "manifest.json"
            shutil.copy2(manifest_src, manifest_dst)

    def _export_protocol_snapshot(self) -> None:
        """Export expanded protocol snapshot."""
        protocol_src = self.run_dir / "protocol_snapshot.json"
        if not protocol_src.exists():
            return

        try:
            with open(protocol_src) as f:
                protocol_data = json.load(f)
        except json.JSONDecodeError:
            return

        # Export as both JSON and formatted text
        protocol_json = self.out_dir / "protocol_snapshot.json"
        with open(protocol_json, "w") as f:
            json.dump(protocol_data, f, indent=2)

        # Create human-readable text version
        protocol_txt = self.out_dir / "protocol_snapshot.txt"
        with open(protocol_txt, "w") as f:
            f.write("PROTOCOL SNAPSHOT\n")
            f.write("=" * 70 + "\n\n")
            self._write_protocol_details(f, protocol_data)

    def _write_protocol_details(self, f, protocol_data: dict) -> None:
        """Write human-readable protocol details."""
        if "name" in protocol_data:
            f.write(f"Name: {protocol_data['name']}\n")
        if "version" in protocol_data:
            f.write(f"Version: {protocol_data['version']}\n")
        f.write("\n")

        if "steps" in protocol_data:
            f.write("PROCESSING STEPS\n")
            f.write("-" * 70 + "\n")
            for i, step in enumerate(protocol_data["steps"], 1):
                f.write(f"\nStep {i}: {step.get('name', 'Unknown')}\n")
                f.write(f"Type: {step.get('type', 'unknown')}\n")
                if step.get("description"):
                    f.write(f"Description: {step['description']}\n")
                if step.get("parameters"):
                    f.write("Parameters:\n")
                    for key, val in step["parameters"].items():
                        f.write(f"  - {key}: {val}\n")
            f.write("\n")

    def _create_environment_freeze(self) -> None:
        """Create environment freeze (pip list)."""
        env_file = self.out_dir / "environment.txt"
        
        try:
            # Try to get pip freeze output
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                with open(env_file, "w") as f:
                    f.write("# Environment freeze generated\n")
                    f.write(f"# Python: {sys.version}\n")
                    f.write(f"# Generated: {datetime.now().isoformat()}\n")
                    f.write("\n")
                    f.write(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: write Python version
            with open(env_file, "w") as f:
                f.write(f"# Python {sys.version}\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")

    def _copy_data_tables(self) -> None:
        """Copy metrics, predictions, and QC tables."""
        tables_dir = self.out_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        # Copy standard artifacts
        for filename in [
            "metrics.json",
            "predictions.json",
            "qc_results.json",
            "uncertainty_metrics.json",
        ]:
            src = self.run_dir / filename
            if src.exists():
                dst = tables_dir / filename
                shutil.copy2(src, dst)

    def _create_plots_index(self) -> None:
        """Create index of all plots in run."""
        plots_dir = self.run_dir / "plots"
        if not plots_dir.exists():
            return

        index_file = self.out_dir / "plots_index.txt"
        with open(index_file, "w") as f:
            f.write("PLOTS INDEX\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Find all plot files
            plot_files = sorted(plots_dir.glob("**/*"))
            for plot_file in plot_files:
                if plot_file.is_file() and plot_file.suffix in [
                    ".png",
                    ".pdf",
                    ".svg",
                    ".jpg",
                ]:
                    relative_path = plot_file.relative_to(plots_dir)
                    file_size = plot_file.stat().st_size
                    f.write(f"- {relative_path} ({file_size} bytes)\n")

    def _create_pack_metadata(self) -> None:
        """Create pack metadata file."""
        metadata = {
            "pack_created": datetime.now().isoformat(),
            "source_run": str(self.run_dir.resolve()),
            "pack_contents": [
                "manifest.json - Execution metadata",
                "protocol_snapshot.json - Protocol configuration (JSON)",
                "protocol_snapshot.txt - Protocol configuration (readable)",
                "environment.txt - Python dependencies",
                "tables/ - Metrics, predictions, QC data",
                "plots_index.txt - Index of generated plots",
                "pack_metadata.json - This file",
            ],
        }

        metadata_file = self.out_dir / "pack_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


class ArchiveExporter:
    """Create stable, shareable archives of analysis runs.
    
    Archives maintain deterministic file ordering for reproducibility
    and support selective inclusion of components.
    """

    def __init__(self) -> None:
        """Initialize exporter."""
        self.run_dir: Path | None = None
        self.include_set: set[str] | None = None

    def export(
        self,
        out_zip_path: str | Path,
        run_dir: str | Path,
        include: Sequence[str] | None = None,
    ) -> Path:
        """Export analysis run to stable archive.
        
        Parameters
        ----------
        out_zip_path : str | Path
            Output zip file path
        run_dir : str | Path
            Source run directory
        include : Sequence[str], optional
            Components to include: "dossier", "figures", "tables", "bundle"
            If None, includes all available components
            
        Returns
        -------
        Path
            Path to created zip file
            
        Raises
        ------
        FileNotFoundError
            If run_dir doesn't exist
        """
        self.run_dir = Path(run_dir)
        out_zip_path = Path(out_zip_path)

        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        # Set include filter
        if include:
            self.include_set = set(include)
        else:
            self.include_set = {"dossier", "figures", "tables", "bundle"}

        # Create archive with deterministic ordering
        out_zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(
            out_zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            self._add_to_archive(zf, self.run_dir)

        return out_zip_path.resolve()

    def _add_to_archive(self, zf: zipfile.ZipFile, base_path: Path) -> None:
        """Add files to archive in deterministic order."""
        # Collect all files to add
        files_to_add = []

        # Critical files first (deterministic order)
        for filename in ["manifest.json", "protocol_snapshot.json"]:
            filepath = base_path / filename
            if filepath.exists():
                files_to_add.append((filepath, filename))

        # Then dossier if included
        if "dossier" in self.include_set:
            dossier_dir = base_path / "dossier"
            if dossier_dir.exists():
                for dossier_file in sorted(dossier_dir.glob("*")):
                    if dossier_file.is_file():
                        arcname = f"dossier/{dossier_file.name}"
                        files_to_add.append((dossier_file, arcname))

        # Then tables if included
        if "tables" in self.include_set:
            for table_file in sorted(base_path.glob("*metrics*.json")):
                arcname = table_file.name
                files_to_add.append((table_file, arcname))
            for table_file in sorted(base_path.glob("*predictions*.json")):
                arcname = table_file.name
                files_to_add.append((table_file, arcname))

        # Then figures if included
        if "figures" in self.include_set:
            plots_dir = base_path / "plots"
            if plots_dir.exists():
                for plot_file in sorted(plots_dir.rglob("*")):
                    if plot_file.is_file():
                        arcname = f"plots/{plot_file.relative_to(plots_dir)}"
                        files_to_add.append((plot_file, arcname))

        # Then bundle if included
        if "bundle" in self.include_set:
            bundle_dir = base_path / "bundle"
            if bundle_dir.exists():
                for bundle_file in sorted(bundle_dir.rglob("*")):
                    if bundle_file.is_file():
                        arcname = f"bundle/{bundle_file.relative_to(bundle_dir)}"
                        files_to_add.append((bundle_file, arcname))

        # Add all files in deterministic order (removing duplicates by arcname)
        seen_arcnames = set()
        for filepath, arcname in sorted(files_to_add, key=lambda x: x[1]):
            if arcname not in seen_arcnames:
                zf.write(filepath, arcname)
                seen_arcnames.add(arcname)


# Public API functions
def build_reproducibility_pack(
    run_dir: str | Path, out_dir: str | Path
) -> Path:
    """Build reproducibility pack from analysis run.
    
    Creates a self-contained pack with:
    - Protocol snapshot (JSON and readable text)
    - Execution manifest
    - Environment freeze (pip list)
    - Data tables (metrics, predictions, QC)
    - Plots index
    
    Parameters
    ----------
    run_dir : str | Path
        Analysis run directory
    out_dir : str | Path
        Output directory for pack
        
    Returns
    -------
    Path
        Path to reproducibility pack directory
        
    Examples
    --------
    >>> pack_dir = build_reproducibility_pack("run_001", "reproducibility_packs")
    """
    builder = ReproducibilityPackBuilder()
    return builder.build(run_dir, out_dir)


def export_archive(
    out_zip_path: str | Path,
    run_dir: str | Path,
    include: Sequence[str] | None = None,
) -> Path:
    """Export analysis run to stable archive.
    
    Creates a zip file with deterministic file ordering for stable,
    reproducible archives.
    
    Parameters
    ----------
    out_zip_path : str | Path
        Output zip file path
    run_dir : str | Path
        Source run directory
    include : Sequence[str], optional
        Components to include:
        - "dossier" : Scientific dossier documents
        - "figures" : Plot files
        - "tables" : Data tables (metrics, predictions, QC)
        - "bundle" : Complete output bundle
        Defaults to all components if not specified.
        
    Returns
    -------
    Path
        Path to created zip archive
        
    Examples
    --------
    >>> archive_path = export_archive(
    ...     "analysis.zip",
    ...     "run_001",
    ...     include=("dossier", "figures", "tables")
    ... )
    """
    exporter = ArchiveExporter()
    return exporter.export(out_zip_path, run_dir, include)


def get_archive_file_list(zip_path: str | Path) -> list[str]:
    """Get deterministically sorted file list from archive.
    
    Useful for verifying archive contents and order.
    
    Parameters
    ----------
    zip_path : str | Path
        Path to zip archive
        
    Returns
    -------
    list[str]
        Sorted list of files in archive
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return []

    with zipfile.ZipFile(zip_path, "r") as zf:
        return sorted(zf.namelist())


def verify_archive_integrity(
    zip_path: str | Path, expected_files: Sequence[str] | None = None
) -> bool:
    """Verify archive integrity and optionally check for expected files.
    
    Parameters
    ----------
    zip_path : str | Path
        Path to zip archive
    expected_files : Sequence[str], optional
        List of files expected to be in archive
        
    Returns
    -------
    bool
        True if archive is valid and contains expected files
    """
    zip_path = Path(zip_path)
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Test archive integrity
            bad_file = zf.testzip()
            if bad_file:
                return False

            # Check expected files if provided
            if expected_files:
                archive_files = set(zf.namelist())
                for expected in expected_files:
                    if expected not in archive_files:
                        return False

        return True
    except (zipfile.BadZipFile, FileNotFoundError):
        return False
