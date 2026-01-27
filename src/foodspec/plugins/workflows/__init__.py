from __future__ import annotations

"""
Registry for workflow plugins.

A workflow can be any callable or lightweight object encapsulating a
multi-step run (e.g., AutoML or harmonization pipelines). The registry stores
these by name for discovery from entry points.
"""


from typing import Dict, Optional


class WorkflowRegistry:
    def __init__(self) -> None:
        self._workflows: Dict[str, object] = {}

    def register(self, name: str, workflow: object, *, override: bool = False) -> None:
        key = name.lower()
        if not override and key in self._workflows:
            raise ValueError(f"Workflow '{name}' already registered")
        self._workflows[key] = workflow

    def get(self, name: str) -> Optional[object]:
        return self._workflows.get(name.lower())

    def unregister(self, name: str) -> None:
        self._workflows.pop(name.lower(), None)

    def available(self):
        return sorted(self._workflows)

    def clear(self) -> None:
        self._workflows.clear()


workflow_registry = WorkflowRegistry()
