from __future__ import annotations

"""
Registry for feature index plugins.

Feature indices can be simple mappings or lightweight classes describing
feature sets. The registry stores them by name for lookup inside workflows
or reporting helpers.
"""


from typing import Dict, Optional


class FeatureIndexRegistry:
    def __init__(self) -> None:
        self._indices: Dict[str, object] = {}

    def register(self, name: str, index: object, *, override: bool = False) -> None:
        key = name.lower()
        if not override and key in self._indices:
            raise ValueError(f"Feature index '{name}' already registered")
        self._indices[key] = index

    def get(self, name: str) -> Optional[object]:
        return self._indices.get(name.lower())

    def unregister(self, name: str) -> None:
        self._indices.pop(name.lower(), None)

    def available(self):
        return sorted(self._indices)

    def clear(self) -> None:
        self._indices.clear()


feature_index_registry = FeatureIndexRegistry()
