"""Processing engine namespace (mindmap-aligned + orchestration)."""

from __future__ import annotations

from .artifacts import (
    Artifact,
    ArtifactRegistry,
    ArtifactType,
    get_registry,
    register_artifact,
    set_registry,
)
from .dag import Node, NodeStatus, NodeType, PipelineDAG, build_standard_pipeline

# Orchestration system
from .orchestrator import ExecutionEngine
from .pipeline import run_preprocessing_pipeline
from .registry import ComponentRegistry

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
