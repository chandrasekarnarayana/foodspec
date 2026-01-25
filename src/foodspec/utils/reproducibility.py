"""Reproducibility helpers for FoodSpec runs."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict

from foodspec.core.run_record import _capture_environment, _capture_seeds


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


def capture_reproducibility() -> Dict[str, Any]:
    """Capture reproducibility metadata for a run."""

    env = _capture_environment()
    seeds = _capture_seeds()
    meta = {
        "environment": env,
        "seeds": seeds,
        "platform": platform.platform(),
        "python_executable": os.getenv("VIRTUAL_ENV", "system"),
        "git_commit": _git_commit_hash(),
    }
    return meta


def write_reproducibility_snapshot(run_dir: str | Path) -> Path:
    """Write reproducibility metadata to run_dir/run_summary.json."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = capture_reproducibility()
    output_path = run_dir / "run_summary.json"
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


__all__ = ["capture_reproducibility", "write_reproducibility_snapshot"]

