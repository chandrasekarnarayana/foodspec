"""Publication-focused helpers for methods text, captions, and checklists."""

from foodspec.report.captions import generate_caption, panel_labels
from foodspec.report.checklist import (
    ChecklistItem,
    checklist_score,
    default_checklist,
    render_checklist,
)
from foodspec.report.journals import get_journal_preset, list_journal_presets
from foodspec.report.methods import MethodsConfig, generate_methods_text, methods_sections
from foodspec.report.stats_notes import statistical_justification, summarize_assumptions

__all__ = [
    "MethodsConfig",
    "generate_methods_text",
    "methods_sections",
    "generate_caption",
    "panel_labels",
    "statistical_justification",
    "summarize_assumptions",
    "ChecklistItem",
    "default_checklist",
    "checklist_score",
    "render_checklist",
    "get_journal_preset",
    "list_journal_presets",
]
