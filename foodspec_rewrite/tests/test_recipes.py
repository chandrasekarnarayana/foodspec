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

from foodspec.core.protocol import ProtocolV2, DataSpec, TaskSpec
from foodspec.preprocess.recipes import resolve_recipe


def test_resolve_recipe_known_and_unknown() -> None:
    steps = resolve_recipe("raman_auth_default")
    assert isinstance(steps, list)
    assert steps[0]["component"] == "DespikeHampel"

    with pytest.raises(ValueError):
        resolve_recipe("unknown")


def test_expand_recipes_uses_builtin(tmp_path) -> None:
    protocol = ProtocolV2(
        data=DataSpec(
            input=str(tmp_path / "data.csv"),
            modality="raman",
            label="target",
            metadata_map={"sample_id": "id", "modality": "mod", "label": "target"},
        ),
        task=TaskSpec(name="classification", objective="max"),
        preprocess={"recipe": "raman_auth_default"},
    )

    expanded = protocol.expand_recipes()
    assert expanded.preprocess.recipe is None
    assert len(expanded.preprocess.steps) >= 2
    assert expanded.preprocess.steps[0].component == "DespikeHampel"
