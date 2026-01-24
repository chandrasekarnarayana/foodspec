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
"""

import pytest

from foodspec.core.pipeline import Pipeline


def test_pipeline_executes_in_topological_order() -> None:
    pipe = Pipeline()
    pipe.add_node("a", lambda ctx: 1)
    pipe.add_node("b", lambda ctx: ctx["a"] + 1, deps=["a"])
    pipe.add_node("c", lambda ctx: ctx["b"] * 2, deps=["b"])

    ctx = pipe.run()
    assert ctx == {"a": 1, "b": 2, "c": 4}


def test_cycle_detection() -> None:
    pipe = Pipeline()
    pipe.add_node("a", lambda ctx: 1, deps=["c"])
    pipe.add_node("b", lambda ctx: 2, deps=["a"])
    pipe.add_node("c", lambda ctx: 3, deps=["b"])

    with pytest.raises(ValueError):
        pipe.run()


def test_missing_dependency() -> None:
    pipe = Pipeline()
    pipe.add_node("a", lambda ctx: 1, deps=["missing"])

    with pytest.raises(ValueError):
        pipe.run()
