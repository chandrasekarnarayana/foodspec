"""Auto-fix suggestion helpers."""

from __future__ import annotations

from typing import Optional

SUGGESTIONS = {
    "file_not_found": "Check the path, ensure the file exists, or rerun with an absolute path.",
    "permission_denied": "Verify filesystem permissions or run with appropriate privileges.",
    "import_error": "Install missing dependencies (pip install -r requirements.txt) and restart.",
    "value_error": "Validate input ranges, types, and required fields before calling the API.",
    "type_error": "Cast or convert inputs to the expected types.",
    "runtime_error": "Re-run with debug logging enabled to capture stack traces.",
}


def suggest_fixes(code: str, context: Optional[str] = None) -> Optional[str]:
    hint = SUGGESTIONS.get(code)
    if context and hint:
        return f"{hint} Context: {context}."
    return hint
