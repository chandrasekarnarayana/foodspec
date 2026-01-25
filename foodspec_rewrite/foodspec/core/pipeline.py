"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

Minimal linear pipeline with DAG-ready topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Set


@dataclass
class PipelineNode:
    """Single pipeline node with optional dependencies.

    Examples
    --------
    >>> node = PipelineNode(name="step1", func=lambda ctx: 1, deps=[])
    >>> node.func({})
    1
    """

    name: str
    func: Callable[[Dict[str, object]], object]
    deps: List[str] = field(default_factory=list)


class Pipeline:
    """Minimal pipeline executor with topological ordering.

    The pipeline runs nodes in dependency order; a future DAG engine can reuse
    the node/edge representation.

    Example
    -------
    >>> pipe = Pipeline()
    >>> pipe.add_node("a", lambda ctx: 1)
    >>> pipe.add_node("b", lambda ctx: ctx["a"] + 1, deps=["a"])
    >>> result = pipe.run()
    >>> result["b"]
    2
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, PipelineNode] = {}

    def add_node(self, name: str, func: Callable[[Dict[str, object]], object], deps: Sequence[str] | None = None) -> None:
        """Register a node; raises if duplicate."""

        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        self._nodes[name] = PipelineNode(name=name, func=func, deps=list(deps or []))

    def _toposort(self) -> List[str]:
        """Return topological order; raise on cycles or missing deps."""

        visited: Set[str] = set()
        temp: Set[str] = set()
        order: List[str] = []

        def dfs(node: str) -> None:
            if node in temp:
                raise ValueError("Cycle detected in pipeline")
            if node in visited:
                return
            if node not in self._nodes:
                raise ValueError(f"Unknown dependency '{node}'")
            temp.add(node)
            for dep in self._nodes[node].deps:
                dfs(dep)
            temp.remove(node)
            visited.add(node)
            order.append(node)

        for name in list(self._nodes):
            if name not in visited:
                dfs(name)
        return order

    def run(self) -> Dict[str, object]:
        """Execute nodes in topological order, returning context."""

        context: Dict[str, object] = {}
        for name in self._toposort():
            node = self._nodes[name]
            context[name] = node.func(context)
        return context


__all__ = ["Pipeline", "PipelineNode"]
