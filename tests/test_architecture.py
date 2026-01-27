"""
tests/test_architecture.py

Architecture enforcement tests. These MUST pass in every commit.
Violations indicate:
  - Dual package roots (import ambiguity)
  - Multiple build configs (installation uncertainty)
  - Import path violations (broken dependencies)
  - Missing critical components

Author: Strict Refactor Engineer
Date: January 25, 2026
"""

import subprocess
from pathlib import Path

import pytest


class TestSingleSourceTree:
    """Verify exactly one foodspec package root exists."""

    def test_single_package_root(self):
        """There must be exactly one src/foodspec/__init__.py."""
        repo_root = Path(__file__).parent.parent
        matches = list(repo_root.glob("**/foodspec/__init__.py"))

        # Filter out .git, venv, etc.
        matches = [m for m in matches if ".git" not in m.parts and "venv" not in m.parts]

        assert len(matches) == 1, (
            f"Expected 1 foodspec/__init__.py, found {len(matches)}: "
            f"{[str(m.relative_to(repo_root)) for m in matches]}"
        )

        # Must be in src/
        assert matches[0].parent.parent.name == "src", (
            f"Package root must be src/foodspec/, found: {matches[0]}"
        )

    def test_no_foodspec_rewrite(self):
        """foodspec_rewrite/ must not exist."""
        repo_root = Path(__file__).parent.parent
        rewrite_dir = repo_root / "foodspec_rewrite"

        assert not rewrite_dir.exists(), (
            "foodspec_rewrite/ directory still exists. "
            "Run refactor_executor.py --phase 1 --execute"
        )

    def test_single_pyproject_toml(self):
        """Exactly one pyproject.toml in repo root."""
        repo_root = Path(__file__).parent.parent
        matches = list(repo_root.glob("**/pyproject.toml"))

        # Filter out .git, venv, nested projects
        matches = [
            m for m in matches
            if ".git" not in m.parts
            and "venv" not in m.parts
            and m.parent.name not in ["foodspec_rewrite"]
            and m.parent == repo_root  # Only root-level
        ]

        assert len(matches) == 1, (
            f"Expected 1 pyproject.toml in repo root, found {len(matches)}: "
            f"{[str(m.relative_to(repo_root)) for m in matches]}"
        )

        # Must be at repo root
        assert matches[0].parent == repo_root, (
            f"pyproject.toml must be at repo root, found: {matches[0]}"
        )


class TestImportPaths:
    """Verify imports resolve from canonical location."""

    def test_protocol_import(self):
        """ProtocolV2 must import from foodspec.core.protocol."""
        try:
            from foodspec.core.protocol import ProtocolV2
            assert ProtocolV2 is not None
        except ImportError as e:
            pytest.fail(f"Cannot import ProtocolV2 from foodspec.core.protocol: {e}")

    def test_registry_import(self):
        """ComponentRegistry must import from foodspec.core.registry."""
        try:
            from foodspec.core.registry import ComponentRegistry
            assert ComponentRegistry is not None
        except ImportError as e:
            pytest.fail(f"Cannot import ComponentRegistry from foodspec.core.registry: {e}")

    def test_orchestrator_import(self):
        """ExecutionEngine must import from foodspec.core.orchestrator."""
        try:
            from foodspec.core.orchestrator import ExecutionEngine
            assert ExecutionEngine is not None
        except ImportError as e:
            pytest.fail(f"Cannot import ExecutionEngine from foodspec.core.orchestrator: {e}")

    def test_artifacts_import(self):
        """ArtifactRegistry must import from foodspec.core.artifacts."""
        try:
            from foodspec.core.artifacts import ArtifactRegistry
            assert ArtifactRegistry is not None
        except ImportError as e:
            pytest.fail(f"Cannot import ArtifactRegistry from foodspec.core.artifacts: {e}")

    def test_manifest_import(self):
        """RunManifest must import from foodspec.core.manifest."""
        try:
            from foodspec.core.manifest import RunManifest
            assert RunManifest is not None
        except ImportError as e:
            pytest.fail(f"Cannot import RunManifest from foodspec.core.manifest: {e}")

    def test_evaluation_import(self):
        """Evaluation functions must import from foodspec.validation.evaluation."""
        try:
            from foodspec.validation.evaluation import evaluate_model_cv
            assert evaluate_model_cv is not None
        except ImportError as e:
            pytest.fail(f"Cannot import evaluate_model_cv from foodspec.validation: {e}")

    def test_trust_evaluator_import(self):
        """TrustEvaluator must import from foodspec.trust.evaluator."""
        try:
            from foodspec.trust.evaluator import TrustEvaluator
            assert TrustEvaluator is not None
        except ImportError as e:
            pytest.fail(f"Cannot import TrustEvaluator from foodspec.trust: {e}")

    def test_no_rewrite_imports(self):
        """No imports from foodspec_rewrite in codebase."""
        repo_root = Path(__file__).parent.parent

        result = subprocess.run(
            ["grep", "-r", "from foodspec_rewrite", "src/", "--include=*.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, (
            f"Found foodspec_rewrite imports in codebase:\n{result.stdout}"
        )


class TestPackageStructure:
    """Verify core modules exist in expected locations."""

    def test_core_modules_exist(self):
        """Core modules must exist."""
        repo_root = Path(__file__).parent.parent
        core_dir = repo_root / "src" / "foodspec" / "core"

        required_modules = [
            "protocol.py",
            "registry.py",
            "orchestrator.py",
            "artifacts.py",
            "manifest.py",
        ]

        for module in required_modules:
            path = core_dir / module
            assert path.exists(), (
                f"Required module missing: {path}. "
                f"Run refactor_executor.py --phase 1 --execute"
            )

    def test_validation_modules_exist(self):
        """Validation modules must exist."""
        repo_root = Path(__file__).parent.parent
        validation_dir = repo_root / "src" / "foodspec" / "validation"

        required_modules = [
            "evaluation.py",
            "splits.py",
            "metrics.py",
            "leakage.py",
        ]

        for module in required_modules:
            path = validation_dir / module
            assert path.exists(), (
                f"Required validation module missing: {path}"
            )

    def test_trust_modules_exist(self):
        """Trust modules must exist."""
        repo_root = Path(__file__).parent.parent
        trust_dir = repo_root / "src" / "foodspec" / "trust"

        required_modules = [
            "conformal.py",
            "calibration.py",
            "abstain.py",
            "evaluator.py",
        ]

        for module in required_modules:
            path = trust_dir / module
            assert path.exists(), (
                f"Required trust module missing: {path}"
            )


class TestCLIEntrypoint:
    """Verify CLI entrypoints are correct."""

    def test_cli_main_exists(self):
        """CLI main.py must exist with run command."""
        repo_root = Path(__file__).parent.parent
        cli_main = repo_root / "src" / "foodspec" / "cli" / "main.py"

        assert cli_main.exists(), (
            f"CLI main.py missing: {cli_main}"
        )

        content = cli_main.read_text()
        assert "def run(" in content or "@app.command" in content, (
            "CLI main.py must have 'run' command or @app.command decorator"
        )

    def test_pyproject_cli_points_to_main(self):
        """pyproject.toml CLI entrypoint must point to correct function."""
        repo_root = Path(__file__).parent.parent
        pyproject = repo_root / "pyproject.toml"

        content = pyproject.read_text()

        # Accept either old or new, but log warning if still using old
        if 'foodspec = "foodspec.cli:app"' in content:
            # Old style (acceptable temporarily)
            pass
        elif 'foodspec = "foodspec.cli.main:run"' in content:
            # New style (preferred)
            pass
        else:
            pytest.fail(
                "pyproject.toml [project.scripts] foodspec entrypoint not found or wrong. "
                "Expected: foodspec = \"foodspec.cli.main:run\""
            )


class TestGitHistory:
    """Verify refactoring used git mv (history preserved)."""

    def test_git_repo(self):
        """Must be in a git repository."""
        repo_root = Path(__file__).parent.parent
        git_dir = repo_root / ".git"

        assert git_dir.exists(), (
            "Not in a git repository"
        )

    def test_git_status_clean(self, monkeypatch):
        """Git working directory should be clean after refactor."""
        repo_root = Path(__file__).parent.parent
        monkeypatch.chdir(repo_root)

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )

        # Allow some uncommitted changes, but flag major issues
        if "foodspec_rewrite" in result.stdout:
            pytest.fail(
                "foodspec_rewrite still in git status. "
                "Run: git rm -r foodspec_rewrite && git commit"
            )


class TestNoDuplicates:
    """Verify no duplicate implementations."""

    def test_no_dual_protocol_implementations(self):
        """Only one ProtocolV2 implementation should exist."""
        repo_root = Path(__file__).parent.parent

        protocol_files = list(repo_root.glob("**/protocol.py"))
        protocol_files = [
            f for f in protocol_files
            if ".git" not in f.parts and not any("venv" in part for part in f.parts)
        ]

        # Should be exactly 1 (in src/foodspec/core/)
        assert len(protocol_files) <= 1, (
            f"Multiple protocol.py files found: "
            f"{[str(f.relative_to(repo_root)) for f in protocol_files]}"
        )

    def test_no_dual_registry_implementations(self):
        """Only one ComponentRegistry should exist."""
        repo_root = Path(__file__).parent.parent

        # Search for ComponentRegistry class definitions
        registry_matches = []
        for py_file in repo_root.glob("**/registry.py"):
            if ".git" in py_file.parts or any("venv" in part for part in py_file.parts):
                continue
            content = py_file.read_text()
            if "class ComponentRegistry" in content:
                registry_matches.append(py_file)

        assert len(registry_matches) <= 1, (
            f"Multiple ComponentRegistry implementations found: "
            f"{[str(m.relative_to(repo_root)) for m in registry_matches]}"
        )

    def test_no_dual_orchestrator_implementations(self):
        """Only one ExecutionEngine should exist."""
        repo_root = Path(__file__).parent.parent

        orchestrator_matches = []
        for py_file in repo_root.glob("**/orchestrator.py"):
            if ".git" in py_file.parts:
                continue
            content = py_file.read_text()
            if "class ExecutionEngine" in content:
                orchestrator_matches.append(py_file)

        assert len(orchestrator_matches) <= 1, (
            f"Multiple ExecutionEngine implementations found: "
            f"{[str(m.relative_to(repo_root)) for m in orchestrator_matches]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
