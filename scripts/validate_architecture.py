#!/usr/bin/env python3
"""
scripts/validate_architecture.py

Validate that refactored architecture is coherent and enforces single-source-tree.

Usage:
    python scripts/validate_architecture.py --strict
    python scripts/validate_architecture.py --full-report > report.txt

Author: Strict Refactor Engineer
Date: January 25, 2026
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


class ArchitectureValidator:
    """Validates FoodSpec architecture coherence."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def log_pass(self, msg: str):
        """Log passing check."""
        print(f"{GREEN}✓{RESET} {msg}")
        self.checks_passed += 1

    def log_fail(self, msg: str):
        """Log failing check."""
        print(f"{RED}✗{RESET} {msg}")
        self.checks_failed += 1

    def log_warn(self, msg: str):
        """Log warning."""
        print(f"{YELLOW}⚠{RESET} {msg}")
        self.warnings += 1

    def log_info(self, msg: str):
        """Log informational message."""
        print(f"{BLUE}ℹ{RESET} {msg}")

    def check_single_package_root(self) -> bool:
        """Verify exactly one foodspec package root."""
        self.log_info("Checking single package root...")
        matches = list(self.repo_root.glob("**/foodspec/__init__.py"))
        matches = [m for m in matches if ".git" not in m.parts and "venv" not in m.parts]

        if len(matches) != 1:
            self.log_fail(
                f"Expected 1 foodspec/__init__.py, found {len(matches)}: "
                f"{[str(m.relative_to(self.repo_root)) for m in matches]}"
            )
            return False

        if matches[0].parent.parent.name != "src":
            self.log_fail(f"Package root not in src/: {matches[0]}")
            return False

        self.log_pass(f"Single package root: {matches[0].relative_to(self.repo_root)}")
        return True

    def check_no_rewrite_directory(self) -> bool:
        """Verify foodspec_rewrite/ does not exist."""
        self.log_info("Checking for foodspec_rewrite directory...")
        rewrite_dir = self.repo_root / "foodspec_rewrite"

        if rewrite_dir.exists():
            self.log_fail("foodspec_rewrite/ directory still exists")
            return False

        self.log_pass("No foodspec_rewrite directory")
        return True

    def check_single_pyproject(self) -> bool:
        """Verify exactly one pyproject.toml."""
        self.log_info("Checking single pyproject.toml...")
        matches = list(self.repo_root.glob("**/pyproject.toml"))
        matches = [
            m for m in matches
            if ".git" not in m.parts
            and "venv" not in m.parts
            and m.parent == self.repo_root
        ]

        if len(matches) != 1:
            self.log_fail(
                f"Expected 1 pyproject.toml, found {len(matches)}: "
                f"{[str(m.relative_to(self.repo_root)) for m in matches]}"
            )
            return False

        self.log_pass("Single pyproject.toml at repo root")
        return True

    def check_critical_imports(self) -> bool:
        """Verify critical imports resolve."""
        self.log_info("Checking critical imports...")
        imports = [
            ("from foodspec.core.protocol import ProtocolV2", "Protocol"),
            ("from foodspec.core.registry import ComponentRegistry", "Registry"),
            ("from foodspec.core.orchestrator import ExecutionEngine", "Orchestrator"),
            ("from foodspec.core.artifacts import ArtifactRegistry", "Artifacts"),
            ("from foodspec.core.manifest import RunManifest", "Manifest"),
            ("from foodspec.validation.evaluation import evaluate_model_cv", "Evaluation"),
            ("from foodspec.trust.evaluator import TrustEvaluator", "Trust"),
        ]

        all_ok = True
        for import_stmt, name in imports:
            result = subprocess.run(
                ["python", "-c", import_stmt],
                cwd=self.repo_root,
                capture_output=True,
            )
            if result.returncode == 0:
                self.log_pass(f"Import {name}")
            else:
                self.log_fail(f"Import {name}")
                all_ok = False

        return all_ok

    def check_core_modules_exist(self) -> bool:
        """Verify core modules exist."""
        self.log_info("Checking core module files...")
        core_dir = self.repo_root / "src" / "foodspec" / "core"

        required = [
            "protocol.py",
            "registry.py",
            "orchestrator.py",
            "artifacts.py",
            "manifest.py",
        ]

        all_ok = True
        for module in required:
            path = core_dir / module
            if path.exists():
                self.log_pass(f"Core module: {module}")
            else:
                self.log_fail(f"Core module missing: {module}")
                all_ok = False

        return all_ok

    def check_no_rewrite_imports(self) -> bool:
        """Verify no imports from foodspec_rewrite."""
        self.log_info("Checking for foodspec_rewrite imports...")

        result = subprocess.run(
            ["grep", "-r", "foodspec_rewrite", "--include=*.py", "src/", "tests/"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and result.stdout:
            self.log_fail("Found imports from foodspec_rewrite")
            print(f"  {result.stdout[:200]}")
            return False

        self.log_pass("No foodspec_rewrite imports")
        return True

    def check_no_duplicate_classes(self) -> bool:
        """Verify no duplicate core class implementations."""
        self.log_info("Checking for duplicate implementations...")

        classes_to_check = [
            ("ProtocolV2", "protocol.py"),
            ("ComponentRegistry", "registry.py"),
            ("ExecutionEngine", "orchestrator.py"),
            ("ArtifactRegistry", "artifacts.py"),
        ]

        all_ok = True
        for class_name, filename in classes_to_check:
            result = subprocess.run(
                ["grep", "-r", f"class {class_name}", "--include=*.py"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )

            # Filter to actual definitions (not comments)
            definitions = [
                line for line in result.stdout.split("\n")
                if f"class {class_name}" in line
                and not line.strip().startswith("#")
                and ".git" not in line
            ]

            if len(definitions) > 1:
                self.log_fail(f"Duplicate {class_name} definitions ({len(definitions)} found)")
                for defn in definitions:
                    print(f"    {defn}")
                all_ok = False
            elif len(definitions) == 1:
                self.log_pass(f"Single {class_name} definition")
            else:
                self.log_fail(f"{class_name} not found")
                all_ok = False

        return all_ok

    def check_git_history(self) -> bool:
        """Verify git history was preserved (used git mv)."""
        self.log_info("Checking git history...")

        result = subprocess.run(
            ["git", "log", "--name-status", "-1"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            self.log_warn("Git history check: verify last commit used git mv (not shown automatically)")
            return True

        return True  # Non-critical if git not available

    def check_cli_entrypoint(self) -> bool:
        """Verify CLI entrypoint is correct."""
        self.log_info("Checking CLI entrypoint...")

        cli_main = self.repo_root / "src" / "foodspec" / "cli" / "main.py"
        if not cli_main.exists():
            self.log_fail("CLI main.py not found")
            return False

        content = cli_main.read_text()
        if "def run(" not in content and "@app.command" not in content:
            self.log_fail("CLI main.py missing run command")
            return False

        self.log_pass("CLI main.py has run command")

        # Check pyproject.toml
        pyproject = self.repo_root / "pyproject.toml"
        if not pyproject.exists():
            self.log_fail("pyproject.toml not found")
            return False

        content = pyproject.read_text()
        if 'foodspec = "foodspec.cli' not in content:
            self.log_fail("pyproject.toml missing foodspec CLI entrypoint")
            return False

        self.log_pass("CLI entrypoint configured in pyproject.toml")
        return True

    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print(f"\n{BLUE}{'=' * 60}{RESET}")
        print(f"{BLUE}FoodSpec Architecture Validation{RESET}")
        print(f"{BLUE}{'=' * 60}{RESET}\n")

        checks = [
            self.check_single_package_root,
            self.check_no_rewrite_directory,
            self.check_single_pyproject,
            self.check_critical_imports,
            self.check_core_modules_exist,
            self.check_no_rewrite_imports,
            self.check_no_duplicate_classes,
            self.check_cli_entrypoint,
            self.check_git_history,
        ]

        for check in checks:
            try:
                check()
            except Exception as e:
                self.log_fail(f"Check failed with exception: {e}")

        print(f"\n{BLUE}{'=' * 60}{RESET}")
        print(f"Results: {GREEN}{self.checks_passed} passed{RESET}, "
              f"{RED}{self.checks_failed} failed{RESET}, "
              f"{YELLOW}{self.warnings} warnings{RESET}")
        print(f"{BLUE}{'=' * 60}{RESET}\n")

        return self.checks_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate FoodSpec architecture coherence"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any check fails",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Generate full report with details",
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    if not (repo_root / ".git").exists():
        print(f"{RED}ERROR{RESET}: Not in a git repository")
        sys.exit(1)

    validator = ArchitectureValidator(repo_root)
    all_passed = validator.run_all_checks()

    if args.strict and not all_passed:
        sys.exit(1)

    sys.exit(1 if (args.strict and not all_passed) else 0)


if __name__ == "__main__":
    main()
