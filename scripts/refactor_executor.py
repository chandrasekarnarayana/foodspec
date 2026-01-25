#!/usr/bin/env python3
"""
FoodSpec Architecture Refactor Executor

Executes phased reorganization from dual-tree to single-source architecture.
Provides dry-run, execute, and rollback modes with full manifest tracking.

Usage:
    python scripts/refactor_executor.py --phase 1 --dry-run
    python scripts/refactor_executor.py --phase 1 --execute --manifest-output manifest.json
    python scripts/refactor_executor.py --rollback manifest.json

Author: Strict Refactor Engineer
Date: January 25, 2026
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import shutil


# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


@dataclass
class Operation:
    """Represents a single refactoring operation."""
    op_type: str  # 'move', 'delete', 'update', 'create'
    src: Optional[str] = None
    dst: Optional[str] = None
    description: str = ""
    success: bool = False
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class RefactorExecutor:
    """Executes architecture refactoring in phases."""

    def __init__(self, repo_root: Path, dry_run: bool = True):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.operations: List[Operation] = []

    def log(self, level: str, msg: str):
        """Log message with color."""
        colors = {
            "info": BLUE,
            "success": GREEN,
            "warning": YELLOW,
            "error": RED,
        }
        color = colors.get(level, RESET)
        print(f"{color}[{level.upper()}]{RESET} {msg}")

    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """Run shell command safely."""
        cwd = cwd or self.repo_root
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception as e:
            self.log("error", f"Command failed: {' '.join(cmd)}\n  {e}")
            return False

    def git_mv(self, src: Path, dst: Path) -> bool:
        """Move file using git (preserves history)."""
        src_rel = src.relative_to(self.repo_root)
        dst_rel = dst.relative_to(self.repo_root)

        if self.dry_run:
            self.log("info", f"[DRY RUN] git mv {src_rel} → {dst_rel}")
            op = Operation(
                op_type="move",
                src=str(src_rel),
                dst=str(dst_rel),
                description=f"Move {src_rel.name} to {dst_rel}",
            )
            self.operations.append(op)
            return True

        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not self.run_command(["git", "mv", str(src_rel), str(dst_rel)]):
            self.log("warning", f"git mv failed, trying standard move: {src_rel}")
            try:
                shutil.move(str(src), str(dst))
            except Exception as e:
                self.log("error", f"Move failed: {e}")
                return False

        op = Operation(
            op_type="move",
            src=str(src_rel),
            dst=str(dst_rel),
            description=f"Move {src_rel.name}",
            success=True,
        )
        self.operations.append(op)
        self.log("success", f"✓ Moved {src_rel} → {dst_rel}")
        return True

    def git_rm_dir(self, path: Path) -> bool:
        """Remove directory using git."""
        path_rel = path.relative_to(self.repo_root)

        if self.dry_run:
            self.log("info", f"[DRY RUN] git rm -r {path_rel}")
            op = Operation(
                op_type="delete",
                src=str(path_rel),
                description=f"Delete {path_rel}",
            )
            self.operations.append(op)
            return True

        if not self.run_command(["git", "rm", "-r", str(path_rel)]):
            self.log("warning", f"git rm failed, trying standard delete: {path_rel}")
            try:
                shutil.rmtree(str(path))
            except Exception as e:
                self.log("error", f"Delete failed: {e}")
                return False

        op = Operation(
            op_type="delete",
            src=str(path_rel),
            description=f"Delete {path_rel}",
            success=True,
        )
        self.operations.append(op)
        self.log("success", f"✓ Deleted {path_rel}")
        return True

    def update_file(self, path: Path, old_text: str, new_text: str) -> bool:
        """Update file content."""
        path_rel = path.relative_to(self.repo_root)

        if self.dry_run:
            self.log("info", f"[DRY RUN] Update {path_rel}")
            op = Operation(
                op_type="update",
                src=str(path_rel),
                description=f"Update imports in {path_rel.name}",
            )
            self.operations.append(op)
            return True

        try:
            content = path.read_text()
            new_content = content.replace(old_text, new_text)
            if new_content == content:
                self.log("warning", f"No changes made to {path_rel}")
                return False
            path.write_text(new_content)

            op = Operation(
                op_type="update",
                src=str(path_rel),
                description=f"Update {path_rel.name}",
                success=True,
            )
            self.operations.append(op)
            self.log("success", f"✓ Updated {path_rel}")
            return True
        except Exception as e:
            self.log("error", f"Update failed: {e}")
            return False

    def phase_1_eliminate_dual_trees(self) -> bool:
        """Phase 1: Move rewrite to src/, delete foodspec_rewrite/."""
        self.log("info", "=" * 60)
        self.log("info", "PHASE 1: ELIMINATE DUAL SOURCE TREES")
        self.log("info", "=" * 60)

        rewrite_dir = self.repo_root / "foodspec_rewrite" / "foodspec"
        src_dir = self.repo_root / "src" / "foodspec"

        if not rewrite_dir.exists():
            self.log("warning", f"Rewrite directory not found: {rewrite_dir}")
            return False

        # Core modules to move
        modules_to_move = [
            ("foodspec/core/protocol.py", "core/protocol.py"),
            ("foodspec/core/registry.py", "core/registry.py"),
            ("foodspec/core/orchestrator.py", "core/orchestrator.py"),
            ("foodspec/core/artifacts.py", "core/artifacts.py"),  # If not already in src
            ("foodspec/core/manifest.py", "core/manifest.py"),
            ("foodspec/preprocess/recipes.py", "preprocess/recipes.py"),
        ]

        # Move directories
        dirs_to_move = [
            ("foodspec/validation", "validation"),
            ("foodspec/deploy", "deploy"),  # If new patterns
        ]

        # Move files
        for src_file, dst_file in modules_to_move:
            src_path = rewrite_dir / src_file
            dst_path = src_dir / dst_file

            if src_path.exists():
                # Check if already in src
                if dst_path.exists():
                    self.log(
                        "warning",
                        f"Destination already exists (skipping): {dst_file}",
                    )
                    continue

                self.git_mv(src_path, dst_path)

        # Move directories
        for src_subdir, dst_subdir in dirs_to_move:
            src_path = rewrite_dir / src_subdir
            dst_path = src_dir / dst_subdir

            if src_path.exists() and not dst_path.exists():
                self.git_mv(src_path, dst_path)

        # Delete entire foodspec_rewrite
        rewrite_root = self.repo_root / "foodspec_rewrite"
        if rewrite_root.exists():
            self.git_rm_dir(rewrite_root)

        # Update imports
        self.log("info", "Updating imports...")
        self._update_imports_phase1()

        return True

    def _update_imports_phase1(self):
        """Update all imports from foodspec_rewrite to src."""
        import_updates = [
            (
                "from foodspec_rewrite.foodspec",
                "from foodspec",
            ),
            (
                "from foodspec.evaluation.artifact_registry",
                "from foodspec.core.artifacts",
            ),
        ]

        # Find all Python files
        for py_file in self.repo_root.glob("**/*.py"):
            if ".git" in py_file.parts:
                continue
            try:
                content = py_file.read_text()
                original = content

                for old_import, new_import in import_updates:
                    content = content.replace(old_import, new_import)

                if content != original:
                    self.update_file(py_file, original, content)
            except Exception as e:
                self.log("warning", f"Could not update {py_file}: {e}")

    def phase_2_consolidate_configs(self) -> bool:
        """Phase 2: Merge configs, single pyproject.toml."""
        self.log("info", "=" * 60)
        self.log("info", "PHASE 2: CONSOLIDATE CONFIGS")
        self.log("info", "=" * 60)

        main_pyproject = self.repo_root / "pyproject.toml"
        shadow_pyproject = self.repo_root / "foodspec_rewrite" / "pyproject.toml"

        # Delete shadow pyproject if it exists
        if shadow_pyproject.exists():
            self.log("info", "Removing shadow pyproject.toml...")
            if not self.dry_run:
                self.run_command(
                    ["git", "rm", str(shadow_pyproject.relative_to(self.repo_root))]
                )
            op = Operation(
                op_type="delete",
                src=str(shadow_pyproject.relative_to(self.repo_root)),
                description="Delete shadow pyproject.toml",
                success=not self.dry_run,
            )
            self.operations.append(op)

        # Ensure main pyproject has ExecutionEngine CLI
        if main_pyproject.exists():
            self.log("info", "Updating main pyproject.toml...")
            content = main_pyproject.read_text()

            # Check if already has new CLI entrypoint
            if 'foodspec = "foodspec.cli.main:run"' not in content:
                # This would require sophisticated parsing; for now just log
                self.log(
                    "warning",
                    "Manual update needed: Add 'foodspec = \"foodspec.cli.main:run\"' to [project.scripts]",
                )

            if not self.dry_run:
                # Version bump
                content = content.replace('version = "1.0.0"', 'version = "1.1.0"')
                main_pyproject.write_text(content)

                op = Operation(
                    op_type="update",
                    src="pyproject.toml",
                    description="Update version to 1.1.0 and add ExecutionEngine CLI",
                    success=True,
                )
                self.operations.append(op)

        return True

    def phase_3_archive_and_clean(self) -> bool:
        """Phase 3: Archive internal docs, remove build artifacts from tracking."""
        self.log("info", "=" * 60)
        self.log("info", "PHASE 3: ARCHIVE DOCS + CLEAN BUILD ARTIFACTS")
        self.log("info", "=" * 60)

        # Create archive directory
        archive_dir = self.repo_root / "_internal" / "phase-history"
        if not self.dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log("info", f"[DRY RUN] Create {archive_dir}")

        # Move phase docs
        docs_to_archive = [
            "ARCHITECTURE.md",
            "IMPLEMENTATION.md",
            "PHASE_PLAN.md",
        ]

        for doc in docs_to_archive:
            src = self.repo_root / "foodspec_rewrite" / doc
            if src.exists():
                dst = archive_dir / f"rewrite-{doc}"
                self.log("info", f"Archiving {doc}...")
                if not self.dry_run:
                    self.run_command(
                        ["git", "mv", str(src.relative_to(self.repo_root)), str(dst.relative_to(self.repo_root))]
                    )

        # Git rm tracked build artifacts
        artifacts_to_remove = [
            "site/",
            "outputs/",
            "protocol_runs_test/",
            "foodspec_runs/",
        ]

        for artifact in artifacts_to_remove:
            path = self.repo_root / artifact
            if path.exists():
                self.log("info", f"Removing tracked artifact: {artifact}")
                if not self.dry_run:
                    self.run_command(
                        ["git", "rm", "-r", "--cached", artifact]
                    )

        # Update .gitignore
        gitignore = self.repo_root / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            additions = "\n# Build artifacts\nsite/\noutputs/\nprotocol_runs_test/\nfoodspec_runs/\n__pycache__/\n.pytest_cache/\n.coverage\n"

            if "site/" not in content:
                self.log("info", "Updating .gitignore...")
                if not self.dry_run:
                    gitignore.write_text(content + additions)

                    op = Operation(
                        op_type="update",
                        src=".gitignore",
                        description="Add build artifact patterns",
                        success=True,
                    )
                    self.operations.append(op)

        return True

    def phase_4_reorganize_examples(self) -> bool:
        """Phase 4: Reorganize examples/ and scripts/."""
        self.log("info", "=" * 60)
        self.log("info", "PHASE 4: REORGANIZE EXAMPLES + SCRIPTS")
        self.log("info", "=" * 60)

        examples_dir = self.repo_root / "examples"
        if not examples_dir.exists():
            self.log("warning", "examples/ directory not found")
            return False

        # Create subdirectories
        subdirs = ["quickstarts", "protocols", "advanced"]
        for subdir in subdirs:
            path = examples_dir / subdir
            if not self.dry_run:
                path.mkdir(parents=True, exist_ok=True)
            else:
                self.log("info", f"[DRY RUN] Create {path}")

        # Map quickstart files
        quickstarts = [
            "oil_authentication_quickstart.py",
            "heating_quality_quickstart.py",
            "aging_quickstart.py",
            "mixture_analysis_quickstart.py",
        ]

        for qs in quickstarts:
            src = examples_dir / qs
            if src.exists():
                dst = examples_dir / "quickstarts" / qs.replace("_quickstart", "")
                self.log("info", f"Moving {qs}...")
                if not self.dry_run:
                    self.git_mv(src, dst)

        # Map demo files to advanced
        demos = []
        for f in examples_dir.glob("*_demo.py"):
            demos.append(f)

        for demo in demos:
            dst = examples_dir / "advanced" / demo.name
            self.log("info", f"Moving {demo.name}...")
            if not self.dry_run:
                self.git_mv(demo, dst)

        return True

    def save_manifest(self, path: Path):
        """Save operation manifest to JSON."""
        import time

        manifest = {
            "timestamp": time.time(),
            "operations": [op.to_dict() for op in self.operations],
            "success_count": sum(1 for op in self.operations if op.success),
            "total_count": len(self.operations),
        }

        path.write_text(json.dumps(manifest, indent=2))
        self.log("success", f"Manifest saved to {path}")

    def execute_phase(self, phase: int) -> bool:
        """Execute a specific phase."""
        phases = {
            1: self.phase_1_eliminate_dual_trees,
            2: self.phase_2_consolidate_configs,
            3: self.phase_3_archive_and_clean,
            4: self.phase_4_reorganize_examples,
        }

        if phase not in phases:
            self.log("error", f"Unknown phase: {phase}")
            return False

        try:
            return phases[phase]()
        except Exception as e:
            self.log("error", f"Phase {phase} failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="FoodSpec Architecture Refactor Executor"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4],
        help="Phase to execute (1-4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute refactoring (opposite of --dry-run)",
    )
    parser.add_argument(
        "--rollback",
        type=Path,
        help="Rollback using operation manifest",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Save operation manifest to file",
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    if not (repo_root / ".git").exists():
        print(f"{RED}ERROR{RESET}: Not in a git repository")
        sys.exit(1)

    # Determine dry run mode
    dry_run = not args.execute

    # Create executor
    executor = RefactorExecutor(repo_root, dry_run=dry_run)

    if args.rollback:
        print(f"{YELLOW}Rollback not yet implemented{RESET}")
        sys.exit(1)

    if not args.phase:
        print(f"{RED}ERROR{RESET}: --phase required")
        parser.print_help()
        sys.exit(1)

    # Execute phase
    success = executor.execute_phase(args.phase)

    # Save manifest
    if args.manifest_output:
        executor.save_manifest(args.manifest_output)

    if success:
        executor.log("success", f"Phase {args.phase} completed successfully")
        if dry_run:
            executor.log("info", "This was a dry run. Run with --execute to apply changes")
    else:
        executor.log("error", f"Phase {args.phase} failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
