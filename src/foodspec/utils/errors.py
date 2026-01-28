"""Human-readable error formatting for user-facing flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from foodspec.utils.hints import suggest_fixes


class FoodSpecValidationError(ValueError):
    """Raised when schema or validation checks fail."""


class FoodSpecQCError(RuntimeError):
    """Raised when QC policy requires a hard failure."""


@dataclass
class FriendlyError:
    code: str
    title: str
    message: str
    suggestion: Optional[str] = None
    severity: str = "error"

    def as_markdown(self) -> str:
        parts = [f"**{self.title}**", self.message]
        if self.suggestion:
            parts.append(f"Hint: {self.suggestion}")
        return "\n".join(parts)


def classify_error(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, ImportError):
        return "import_error"
    if isinstance(exc, ValueError):
        return "value_error"
    if isinstance(exc, TypeError):
        return "type_error"
    return "runtime_error"


def friendly_error(exc: Exception, context: Optional[str] = None) -> FriendlyError:
    code = classify_error(exc)
    base_msg = str(exc) or code.replace("_", " ")

    title_map = {
        "file_not_found": "Missing file",
        "permission_denied": "Permission denied",
        "import_error": "Dependency import failed",
        "value_error": "Invalid value",
        "type_error": "Type mismatch",
        "runtime_error": "Unexpected runtime error",
    }

    title = title_map.get(code, "Error")
    message = base_msg if context is None else f"{base_msg} (while {context})"
    suggestion = suggest_fixes(code, context=context)

    return FriendlyError(code=code, title=title, message=message, suggestion=suggestion)
