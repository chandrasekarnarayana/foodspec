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

from foodspec.core.registry import ComponentRegistry


class DummyPreprocess:
    def __init__(self, method: str) -> None:
        self.method = method


class DummyModel:
    def __init__(self, C: float = 1.0) -> None:
        self.C = C


def test_register_and_create_success() -> None:
    registry = ComponentRegistry()
    registry.register("preprocess", "normalize", DummyPreprocess)
    registry.register("model", "logreg", DummyModel)

    step = registry.create("preprocess", "normalize", method="area")
    model = registry.create("model", "logreg", C=0.5)

    assert isinstance(step, DummyPreprocess)
    assert step.method == "area"
    assert isinstance(model, DummyModel)
    assert model.C == 0.5


def test_unknown_component_lists_available() -> None:
    registry = ComponentRegistry()
    registry.register("preprocess", "normalize", DummyPreprocess)

    with pytest.raises(ValueError) as err:
        registry.create("preprocess", "smooth")

    msg = str(err.value)
    assert "Unknown component" in msg
    assert "normalize" in msg  # lists available


def test_duplicate_registration_error() -> None:
    registry = ComponentRegistry()
    registry.register("preprocess", "normalize", DummyPreprocess)

    with pytest.raises(ValueError):
        registry.register("preprocess", "normalize", DummyPreprocess)


def test_unknown_category_error() -> None:
    registry = ComponentRegistry()

    with pytest.raises(ValueError):
        registry.register("unknown", "x", DummyPreprocess)

    with pytest.raises(ValueError):
        registry.available("unknown")

    with pytest.raises(ValueError):
        registry.create("unknown", "x")
