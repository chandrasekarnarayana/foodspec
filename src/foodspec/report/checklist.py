"""Reproducibility checklist helpers."""
from __future__ import annotations


from dataclasses import dataclass
from typing import List


@dataclass
class ChecklistItem:
    name: str
    status: bool = False
    notes: str = ""


def default_checklist() -> List[ChecklistItem]:
    """Return a baseline reproducibility checklist."""

    return [
        ChecklistItem("Data availability statement"),
        ChecklistItem("Instrument settings documented"),
        ChecklistItem("Preprocessing parameters versioned"),
        ChecklistItem("Model hyperparameters and seeds stored"),
        ChecklistItem("Evaluation splits fixed and archived"),
        ChecklistItem("Statistical tests and corrections listed"),
        ChecklistItem("Code and environment exportable"),
        ChecklistItem("External validation attempted", notes="State if not applicable"),
    ]


def render_checklist(items: List[ChecklistItem]) -> str:
    """Render checklist as Markdown with checkboxes."""

    lines = []
    for item in items:
        mark = "x" if item.status else " "
        note = f" ({item.notes})" if item.notes else ""
        lines.append(f"- [{mark}] {item.name}{note}")
    return "\n".join(lines)


def checklist_score(items: List[ChecklistItem]) -> float:
    """Return completion ratio between 0 and 1."""

    if not items:
        return 0.0
    completed = sum(1 for item in items if item.status)
    return completed / len(items)
