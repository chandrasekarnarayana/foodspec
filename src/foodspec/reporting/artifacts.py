"""Reporting artifact helpers built on the FoodSpec run artifact contract."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, Optional

from foodspec.core.run_record import _capture_environment, _capture_seeds
from foodspec.utils.run_artifacts import (
    get_logger,
    init_run_dir,
    write_manifest,
    write_run_summary,
)


def _cpu_info() -> str:
    cpu = platform.processor()
    if cpu:
        return cpu
    return platform.uname().machine


def init_reporting_run(
    out_dir: Path | str,
    *,
    command: str,
    inputs: list[Path | str],
    config: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
    protocol_path: Optional[Path | str] = None,
    args: Optional[list[str]] = None,
) -> Path:
    """Initialize a reporting run directory and write base manifest."""
    run_dir = init_run_dir(out_dir)
    get_logger(run_dir)
    env = _capture_environment()
    seeds = _capture_seeds()
    if seed is not None:
        seeds["cli_seed"] = int(seed)
    manifest_payload: Dict[str, Any] = {
        "command": command,
        "command_args": args or [],
        "inputs": inputs,
        "mode": mode,
        "seed": seed,
        "cpu": _cpu_info(),
        "environment": env,
        "random_seeds": seeds,
        "config": config or {},
    }
    if protocol_path:
        manifest_payload["protocol_path"] = str(protocol_path)
    write_manifest(run_dir, manifest_payload)
    return run_dir


def finalize_reporting_run(run_dir: Path, summary: Dict[str, Any]) -> None:
    """Write run_summary.json for reporting workflows."""
    write_run_summary(run_dir, summary)


__all__ = ["init_reporting_run", "finalize_reporting_run"]
