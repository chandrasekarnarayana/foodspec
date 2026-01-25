"""Processing engine namespace (mindmap-aligned)."""
from __future__ import annotations

from .pipeline import run_preprocessing_pipeline
from .registry import ComponentRegistry

__all__ = ["ComponentRegistry", "run_preprocessing_pipeline"]

