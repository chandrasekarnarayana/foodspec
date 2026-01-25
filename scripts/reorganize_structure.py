#!/usr/bin/env python3
"""
File Structure Reorganization Script for FoodSpec

This script automates the repository reorganization outlined in FILE_STRUCTURE_AUDIT.md.

Usage:
    python scripts/reorganize_structure.py --dry-run  # Preview changes
    python scripts/reorganize_structure.py --execute  # Execute changes

Author: GitHub Copilot
Date: January 25, 2026
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


class ReorganizationExecutor:
    """Executes repository reorganization safely."""

    def __init__(self, repo_root: Path, dry_run: bool = True):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.actions_log: List[Tuple[str, str]] = []

    def log_action(self, action: str, path: str, success: bool = True):
        """Log an action for review."""
        status = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
        print(f"{status} {action}: {path}")
        self.actions_log.append((action, path))

    def create_directory(self, path: Path):
        """Create a directory if it doesn't exist."""
        if self.dry_run:
            self.log_action("CREATE DIR", str(path))
            return

        try:
            path.mkdir(parents=True, exist_ok=True)
            self.log_action("CREATE DIR", str(path))
        except Exception as e:
            self.log_action("CREATE DIR", str(path), success=False)
            print(f"  {RED}Error: {e}{RESET}")

    def move_file(self, src: Path, dst: Path):
        """Move a file with git."""
        if not src.exists():
            print(f"  {YELLOW}Warning: Source doesn't exist: {src}{RESET}")
            return

        if self.dry_run:
            self.log_action("MOVE", f"{src} → {dst}")
            return

        try:
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Use git mv for tracked files
            result = subprocess.run(
                ["git", "mv", str(src), str(dst)],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Fallback to regular move for untracked files
                shutil.move(src, dst)
            
            self.log_action("MOVE", f"{src} → {dst}")
        except Exception as e:
            self.log_action("MOVE", f"{src} → {dst}", success=False)
            print(f"  {RED}Error: {e}{RESET}")

    def remove_directory(self, path: Path):
        """Remove a directory with git."""
        if not path.exists():
            print(f"  {YELLOW}Warning: Path doesn't exist: {path}{RESET}")
            return

        if self.dry_run:
            self.log_action("REMOVE DIR", str(path))
            return

        try:
            # Use git rm -r for tracked directories
            result = subprocess.run(
                ["git", "rm", "-r", str(path)],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Fallback to regular removal for untracked
                shutil.rmtree(path)
            
            self.log_action("REMOVE DIR", str(path))
        except Exception as e:
            self.log_action("REMOVE DIR", str(path), success=False)
            print(f"  {RED}Error: {e}{RESET}")

    def update_gitignore(self):
        """Update .gitignore with new entries."""
        gitignore_additions = """
# Build outputs
site/
dist/
build/
*.egg-info/

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ruff_cache/
.coverage
.coverage.*
htmlcov/
.benchmarks/

# Application cache
.foodspec_cache/

# Demo/test outputs
outputs/
demo_*/
comparison_output/
protocol_runs_test/
*_runs/
*_output/
*_export/

# Temporary files
*.tmp
*.log
"""

        gitignore_path = self.repo_root / ".gitignore"
        
        if self.dry_run:
            self.log_action("UPDATE", ".gitignore")
            return

        try:
            # Read existing content
            if gitignore_path.exists():
                with open(gitignore_path, "r") as f:
                    existing = f.read()
            else:
                existing = ""

            # Only add if not already present
            if "# Demo/test outputs" not in existing:
                with open(gitignore_path, "a") as f:
                    f.write(gitignore_additions)
                self.log_action("UPDATE", ".gitignore")
            else:
                print(f"  {YELLOW}Info: .gitignore already updated{RESET}")
        except Exception as e:
            self.log_action("UPDATE", ".gitignore", success=False)
            print(f"  {RED}Error: {e}{RESET}")

    def clean_ignored_files(self):
        """Clean files that are now ignored."""
        if self.dry_run:
            self.log_action("CLEAN", "ignored files (git clean -fdX)")
            return

        try:
            result = subprocess.run(
                ["git", "clean", "-fdX"],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            self.log_action("CLEAN", "ignored files")
            if result.stdout:
                print(f"  {BLUE}Removed: {result.stdout.strip()}{RESET}")
        except Exception as e:
            self.log_action("CLEAN", "ignored files", success=False)
            print(f"  {RED}Error: {e}{RESET}")

    def step1_update_gitignore(self):
        """Step 1: Update .gitignore."""
        print(f"\n{BLUE}=== Step 1: Update .gitignore ==={RESET}")
        self.update_gitignore()

    def step2_clean_ignored(self):
        """Step 2: Clean ignored files."""
        print(f"\n{BLUE}=== Step 2: Clean Ignored Files ==={RESET}")
        self.clean_ignored_files()

    def step3_archive_phase_docs(self):
        """Step 3: Archive phase documents."""
        print(f"\n{BLUE}=== Step 3: Archive Phase Documents ==={RESET}")
        
        # Create archive structure
        archive_root = self.repo_root / "_internal" / "phase-history"
        self.create_directory(archive_root / "phase-1-8")
        self.create_directory(archive_root / "architecture-docs")
        self.create_directory(archive_root / "joss-prep")

        # Move root-level phase documents
        phase_files = [
            "PHASE10_PROMPT17_COMPLETION.md",
            "PHASE11_REPORTING_COMPLETION.md",
            "PHASE12_PAPER_PRESETS_COMPLETION.md",
            "PHASE13_DOSSIER_COMPLETION.md",
            "PHASE_14_SUMMARY.md",
            "PHASE_1_IMPLEMENTATION.md",
            "PHASE_7_SUMMARY.md",
            "PHASE8_COMPLETION_REPORT.md",
            "PHASE9_COMPLETION_REPORT.md"
        ]

        for filename in phase_files:
            src = self.repo_root / filename
            dst = archive_root / "phase-1-8" / filename
            self.move_file(src, dst)

        # Move JOSS documents
        joss_files = [
            "JOSS_DOCS_AUDIT_REPORT.md",
            "JOSS_SUBMISSION_CHECKLIST.md"
        ]

        for filename in joss_files:
            src = self.repo_root / filename
            dst = archive_root / "joss-prep" / filename
            if src.exists():
                self.move_file(src, dst)

        # Create README index
        readme_content = """# FoodSpec Implementation Phase History

Complete record of the 8-phase rewrite from v1.0.0 to v1.1.0.

## Directory Structure

- `phase-1-8/` - Numbered phase completion reports
- `architecture-docs/` - Technical architecture documentation
- `joss-prep/` - JOSS submission preparation documents

## Migration

See `BRANCH_MIGRATION_PLAN.md` in project root for the v1.0 → v2.0 migration strategy.

## Phases Summary

1. **Phase 1**: Trust Subsystem & Core API
2. **Phase 7**: Protocol System
3. **Phase 8**: Enhanced QC Engine
4. **Phase 9**: Validation Framework
5. **Phase 10**: Interpretability
6. **Phase 11**: Reporting Infrastructure
7. **Phase 12**: Paper Presets
8. **Phase 13**: Dossier Generation
9. **Phase 14**: Final Integration

## Timeline

- **Start**: December 2025
- **v1.1.0-rc1**: January 25, 2026
- **v1.1.0 stable**: February 2026 (planned)
- **v2.0.0**: July 2026 (planned)
"""

        readme_path = archive_root / "README.md"
        if self.dry_run:
            self.log_action("CREATE", str(readme_path))
        else:
            try:
                with open(readme_path, "w") as f:
                    f.write(readme_content)
                self.log_action("CREATE", str(readme_path))
            except Exception as e:
                self.log_action("CREATE", str(readme_path), success=False)
                print(f"  {RED}Error: {e}{RESET}")

    def step4_archive_foodspec_rewrite_docs(self):
        """Step 4: Archive foodspec_rewrite/ documentation."""
        print(f"\n{BLUE}=== Step 4: Archive foodspec_rewrite/ Docs ==={RESET}")
        
        rewrite_dir = self.repo_root / "foodspec_rewrite"
        if not rewrite_dir.exists():
            print(f"  {YELLOW}Info: foodspec_rewrite/ already removed{RESET}")
            return

        archive_root = self.repo_root / "_internal" / "phase-history"
        arch_docs = archive_root / "architecture-docs"

        # Move markdown files from foodspec_rewrite/
        md_files = list(rewrite_dir.glob("*.md"))
        for src in md_files:
            dst = arch_docs / src.name
            self.move_file(src, dst)

        # Move docs/ markdown files
        docs_dir = rewrite_dir / "docs"
        if docs_dir.exists():
            doc_md_files = list(docs_dir.glob("*.md"))
            for src in doc_md_files:
                dst = arch_docs / src.name
                self.move_file(src, dst)

    def step5_remove_foodspec_rewrite(self):
        """Step 5: Remove foodspec_rewrite/ directory."""
        print(f"\n{BLUE}=== Step 5: Remove foodspec_rewrite/ ==={RESET}")
        
        rewrite_dir = self.repo_root / "foodspec_rewrite"
        if rewrite_dir.exists():
            self.remove_directory(rewrite_dir)
        else:
            print(f"  {YELLOW}Info: foodspec_rewrite/ already removed{RESET}")

    def step6_reorganize_examples(self):
        """Step 6: Reorganize examples/ directory."""
        print(f"\n{BLUE}=== Step 6: Reorganize examples/ ==={RESET}")
        
        examples_dir = self.repo_root / "examples"
        if not examples_dir.exists():
            print(f"  {YELLOW}Warning: examples/ doesn't exist{RESET}")
            return

        # Create subdirectories
        subdirs = ["quickstarts", "advanced", "validation", "new-features"]
        for subdir in subdirs:
            self.create_directory(examples_dir / subdir)

        # File categorization
        quickstarts = [
            "oil_authentication_quickstart.py",
            "heating_quality_quickstart.py",
            "mixture_analysis_quickstart.py",
            "aging_quickstart.py",
            "phase1_quickstart.py",
            "qc_quickstart.py"
        ]

        advanced = [
            "foodspec_auto_analysis_script.py",
            "governance_demo.py",
            "hyperspectral_demo.py",
            "moats_demo.py",
            "multimodal_fusion_demo.py",
            "spectral_dataset_demo.py",
            "vip_demo.py",
            "foodspec_rq_demo.py"
        ]

        validation = [
            "validation_chemometrics_oils.py",
            "validation_peak_ratios.py",
            "validation_preprocessing_baseline.py"
        ]

        new_features = [
            "multi_run_comparison_demo.py",
            "uncertainty_demo.py",
            "export_demo.py",
            "pdf_export_demo.py",
            "paper_presets_demo.py",
            "embeddings_demo.py",
            "processing_stages_demo.py",
            "coefficients_stability_demo.py"
        ]

        # Move files
        for filename in quickstarts:
            src = examples_dir / filename
            dst = examples_dir / "quickstarts" / filename
            if src.exists():
                self.move_file(src, dst)

        for filename in advanced:
            src = examples_dir / filename
            dst = examples_dir / "advanced" / filename
            if src.exists():
                self.move_file(src, dst)

        for filename in validation:
            src = examples_dir / filename
            dst = examples_dir / "validation" / filename
            if src.exists():
                self.move_file(src, dst)

        for filename in new_features:
            src = examples_dir / filename
            dst = examples_dir / "new-features" / filename
            if src.exists():
                self.move_file(src, dst)

    def step7_reorganize_scripts(self):
        """Step 7: Reorganize scripts/ directory."""
        print(f"\n{BLUE}=== Step 7: Reorganize scripts/ ==={RESET}")
        
        scripts_dir = self.repo_root / "scripts"
        if not scripts_dir.exists():
            print(f"  {YELLOW}Warning: scripts/ doesn't exist{RESET}")
            return

        # Create subdirectories
        subdirs = ["development", "documentation", "maintenance", "workflows"]
        for subdir in subdirs:
            self.create_directory(scripts_dir / subdir)

        # File categorization
        development = [
            "audit_imports.py",
            "test_examples_imports.py",
            "execute_migration.py"
        ]

        documentation = [
            "generate_docs_figures.py",
            "generate_workflow_figure.py",
            "validate_docs.py",
            "check_docs_links.py",
            "bulk_update_links.py"
        ]

        maintenance = [
            "fix_codeblock_languages.py",
            "fix_methods_depth.py",
            "fix_tutorials_depth.py",
            "fix_workflows_depth.py"
        ]

        workflows = [
            "raman_workflow_foodspec.py"
        ]

        # Move files
        for filename in development:
            src = scripts_dir / filename
            dst = scripts_dir / "development" / filename
            if src.exists():
                self.move_file(src, dst)

        for filename in documentation:
            src = scripts_dir / filename
            dst = scripts_dir / "documentation" / filename
            if src.exists():
                self.move_file(src, dst)

        for filename in maintenance:
            src = scripts_dir / filename
            dst = scripts_dir / "maintenance" / filename
            if src.exists():
                self.move_file(src, dst)

        for filename in workflows:
            src = scripts_dir / filename
            dst = scripts_dir / "workflows" / filename
            if src.exists():
                self.move_file(src, dst)

    def generate_summary(self):
        """Generate summary of all actions."""
        print(f"\n{BLUE}=== Reorganization Summary ==={RESET}")
        print(f"\nTotal actions: {len(self.actions_log)}")
        
        action_types = {}
        for action, _ in self.actions_log:
            action_types[action] = action_types.get(action, 0) + 1
        
        for action, count in sorted(action_types.items()):
            print(f"  {action}: {count}")

        if self.dry_run:
            print(f"\n{YELLOW}⚠ DRY RUN: No changes were made{RESET}")
            print(f"Run with --execute to apply changes")
        else:
            print(f"\n{GREEN}✓ Changes applied successfully{RESET}")
            print(f"Review changes and commit when ready")

    def execute(self):
        """Execute all reorganization steps."""
        print(f"{BLUE}{'=' * 60}{RESET}")
        print(f"{BLUE}FoodSpec Repository Reorganization{RESET}")
        print(f"{BLUE}{'=' * 60}{RESET}")
        
        if self.dry_run:
            print(f"{YELLOW}Running in DRY RUN mode{RESET}")
        else:
            print(f"{RED}EXECUTING CHANGES{RESET}")
            response = input(f"\n{YELLOW}Are you sure? This will modify files. (yes/no): {RESET}")
            if response.lower() != "yes":
                print(f"{RED}Aborted{RESET}")
                return

        try:
            self.step1_update_gitignore()
            self.step2_clean_ignored()
            self.step3_archive_phase_docs()
            self.step4_archive_foodspec_rewrite_docs()
            self.step5_remove_foodspec_rewrite()
            self.step6_reorganize_examples()
            self.step7_reorganize_scripts()
            self.generate_summary()

        except KeyboardInterrupt:
            print(f"\n{RED}Interrupted by user{RESET}")
            sys.exit(1)
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize FoodSpec repository structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/reorganize_structure.py --dry-run   # Preview changes
  python scripts/reorganize_structure.py --execute   # Apply changes
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    group.add_argument(
        "--execute",
        action="store_true",
        help="Execute reorganization (CAUTION: modifies files)"
    )

    args = parser.parse_args()

    # Get repository root
    repo_root = Path(__file__).parent.parent
    if not (repo_root / ".git").exists():
        print(f"{RED}Error: Not in a git repository{RESET}")
        sys.exit(1)

    # Execute reorganization
    executor = ReorganizationExecutor(repo_root, dry_run=args.dry_run)
    executor.execute()


if __name__ == "__main__":
    main()
