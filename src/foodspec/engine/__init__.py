"""Processing engine namespace (mindmap-aligned + orchestration)."""
from __future__ import annotations

from .pipeline import run_preprocessing_pipeline
from .registry import ComponentRegistry

# Orchestration system
from .orchestrator import ExecutionEngine
from .dag import PipelineDAG, Node, NodeType, NodeStatus, build_standard_pipeline
from .artifacts import (
    ArtifactRegistry,
    Artifact,
    ArtifactType,
    get_registry,
    set_registry,
    register_artifact,
)

__all__ = [
    # Existing
    "ComponentRegistry",
    "run_preprocessing_pipeline",
    # New orchestration
    "ExecutionEngine",
    "PipelineDAG",
    "Node",
    "NodeType",
    "NodeStatus",
    "build_standard_pipeline",
    "ArtifactRegistry",
    "Artifact",
    "ArtifactType",
    "get_registry",
    "set_registry",
    "register_artifact",
]

