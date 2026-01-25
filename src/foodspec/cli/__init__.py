from __future__ import annotations
"""Command-line interface for foodspec.

This module provides the main CLI entry point.
"""


# Import the main CLI app from main.py
from foodspec.cli.main import app

# Backward-compatibility re-export for tests that monkeypatch FoodSpec
from foodspec.core.api import FoodSpec

__all__ = ["app", "FoodSpec"]
