"""
Canonical registry module for foodspec.core.

This module provides the public API for component registration and lookup.
It re-exports or provides default implementations for a clean, canonical location.
"""

from __future__ import annotations

# Try to import from foodspec.registry if ComponentRegistry exists there
try:
    from foodspec.registry import ComponentRegistry
except ImportError:
    # If not available, provide a default implementation
    class ComponentRegistry:
        """Registry for component discovery and instantiation."""

        _registry = {}

        @classmethod
        def register(cls, name: str, component_class):
            """Register a component."""
            cls._registry[name] = component_class

        @classmethod
        def get(cls, name: str):
            """Get a registered component."""
            return cls._registry.get(name)

        @classmethod
        def list_components(cls):
            """List all registered components."""
            return list(cls._registry.keys())


__all__ = [
    "ComponentRegistry",
]
