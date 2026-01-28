"""Diff utilities for comparing FoodSpec run records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from foodspec.core.run_record import RunRecord


def _load_record(run: Union[RunRecord, str, Path]) -> RunRecord:
    if isinstance(run, RunRecord):
        return run
    return RunRecord.from_json(run)


def _dict_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    keys = set(a.keys()).union(b.keys())
    for key in sorted(keys):
        if a.get(key) != b.get(key):
            diff[key] = {"run_a": a.get(key), "run_b": b.get(key)}
    return diff


def diff_runs(run_a: Union[RunRecord, str, Path], run_b: Union[RunRecord, str, Path]) -> Dict[str, Any]:
    """Compare two runs and summarize changes and likely effects."""

    rec_a = _load_record(run_a)
    rec_b = _load_record(run_b)

    changes: List[Dict[str, Any]] = []
    expected: List[str] = []

    def _add_change(field: str, val_a: Any, val_b: Any, consequence: str | None) -> None:
        if val_a != val_b:
            changes.append({"field": field, "run_a": val_a, "run_b": val_b})
            if consequence:
                expected.append(consequence)

    _add_change("dataset_hash", rec_a.dataset_hash, rec_b.dataset_hash, "Dataset changed; model metrics may shift.")
    _add_change(
        "config_hash", rec_a.config_hash, rec_b.config_hash, "Configuration changed; pipeline behavior differs."
    )

    if rec_a.step_records != rec_b.step_records:
        changes.append({"field": "step_records", "run_a": rec_a.step_records, "run_b": rec_b.step_records})
        expected.append("Pipeline steps or parameters differ.")

    env_diff = _dict_diff(rec_a.environment, rec_b.environment)
    if env_diff:
        changes.append({"field": "environment", "diff": env_diff})
        expected.append("Environment drift may impact numerical reproducibility.")

    seeds_diff = _dict_diff(rec_a.random_seeds, rec_b.random_seeds)
    if seeds_diff:
        changes.append({"field": "random_seeds", "diff": seeds_diff})
        expected.append("Random seeds differ; stochastic components may diverge.")

    if rec_a.output_paths != rec_b.output_paths:
        changes.append({"field": "output_paths", "run_a": rec_a.output_paths, "run_b": rec_b.output_paths})

    expected_unique = list(dict.fromkeys(expected))

    return {
        "run_a": rec_a.run_id,
        "run_b": rec_b.run_id,
        "changes": changes,
        "expected_consequences": expected_unique,
        "text": _format_diff_text(rec_a.run_id, rec_b.run_id, changes, expected_unique),
    }


def _format_diff_text(run_a_id: str, run_b_id: str, changes: List[Dict[str, Any]], expected: List[str]) -> str:
    lines = [f"diff_runs: {run_a_id} vs {run_b_id}"]
    if not changes:
        lines.append("No differences detected.")
    else:
        lines.append("Changes:")
        for change in changes:
            field = change["field"]
            if "diff" in change:
                lines.append(f"- {field}: keys differ")
            else:
                lines.append(f"- {field}: {change.get('run_a')} -> {change.get('run_b')}")
    if expected:
        lines.append("Expected consequences:")
        for item in expected:
            lines.append(f"- {item}")
    return "\n".join(lines)


__all__ = ["diff_runs"]
