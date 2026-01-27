"""Run artifact utilities for CLI and pipeline workflows."""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from foodspec._version import __version__
from foodspec.logging_utils import get_logger as _get_logger
from foodspec.logging_utils import setup_logging

_DEFAULT_MAX_HASH_MB = 200


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit() -> Optional[str]:
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


def _hash_file(path: Path, chunk_size: int = 8192) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_inputs(inputs: Iterable[Path | str]) -> list[Dict[str, Any]]:
    payload: list[Dict[str, Any]] = []
    for item in inputs:
        p = Path(item)
        entry: Dict[str, Any] = {"path": str(p)}
        if p.exists() and p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            entry["size_mb"] = round(size_mb, 3)
            if size_mb <= _DEFAULT_MAX_HASH_MB:
                entry["sha256"] = _hash_file(p)
            else:
                entry["sha256"] = None
                entry["hash_skipped"] = f"> {_DEFAULT_MAX_HASH_MB} MB"
        payload.append(entry)
    return payload


def init_run_dir(outdir: Optional[Path | str]) -> Path:
    """Initialize run directory with logs/run.log configured."""
    if outdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        outdir = Path("runs") / f"run_{stamp}"
    run_dir = Path(outdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir=logs_dir)
    return run_dir


def get_logger(run_dir: Optional[Path | str]) -> logging.Logger:
    """Return a configured logger (writes to logs/run.log when run_dir provided)."""
    if run_dir is None:
        return _get_logger()
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return setup_logging(run_dir=logs_dir)


def safe_json_dump(path: Path, obj: Dict[str, Any]) -> Path:
    """Write JSON to disk, ensuring parent directories exist."""
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize(obj), indent=2))
    return path


def write_run_summary(run_dir: Path | str, summary: Dict[str, Any]) -> Path:
    """Merge and write run_summary.json."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    payload: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text())
        except Exception:
            payload = {}
    payload.update(summary)
    return safe_json_dump(summary_path, payload)


def write_manifest(run_dir: Path | str, manifest: Dict[str, Any]) -> Path:
    """Write manifest.json with base environment fields and provided metadata."""
    run_dir = Path(run_dir)
    payload = dict(manifest)
    base: Dict[str, Any] = {
        "timestamp": _now_utc(),
        "foodspec_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }

    inputs = payload.get("inputs") or payload.get("input_paths")
    if inputs and isinstance(inputs, (list, tuple)):
        base["inputs"] = _hash_inputs(inputs)
        payload["inputs"] = base["inputs"]
        payload.pop("input_paths", None)

    protocol_path = payload.get("protocol_path")
    if protocol_path:
        protocol_file = Path(protocol_path)
        base["protocol_path"] = str(protocol_file)
        if protocol_file.exists() and protocol_file.is_file():
            base["protocol_sha256"] = _hash_file(protocol_file)
        payload["protocol_path"] = str(protocol_file)

    base.update(payload)
    return safe_json_dump(run_dir / "manifest.json", base)


def update_manifest(run_dir: Path | str, updates: Dict[str, Any]) -> Path:
    """Merge updates into manifest.json without dropping existing fields."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return write_manifest(run_dir, updates)
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        payload = {}
    payload.update(updates)
    return safe_json_dump(manifest_path, payload)


__all__ = [
    "init_run_dir",
    "write_manifest",
    "update_manifest",
    "write_run_summary",
    "get_logger",
    "safe_json_dump",
]
