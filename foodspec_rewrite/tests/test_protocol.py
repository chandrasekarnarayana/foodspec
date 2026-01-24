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

from foodspec.core.protocol import (
    DataSpec,
    PreprocessSpec,
    PreprocessStep,
    ProtocolV2,
    TaskSpec,
)


def test_validate_missing_metadata_keys_actionable_error() -> None:
    """Validation should fail with actionable message when metadata_map is incomplete."""

    protocol = ProtocolV2(
        data=DataSpec(
            input="data.csv",
            modality="raman",
            label="target",
            metadata_map={"label": "target", "modality": "modality"},
        ),
        task=TaskSpec(name="classification", objective="maximize accuracy"),
    )

    with pytest.raises(ValueError) as err:
        protocol.validate()

    assert "metadata_map" in str(err.value)
    assert "sample_id" in str(err.value)


def test_expand_recipes_and_unknown_components() -> None:
    """Recipe expansion should materialize steps and flag unknown components."""

    recipe_registry = {
        "basic": [
            {"component": "normalize", "params": {"method": "area"}},
            {"component": "smooth", "params": {"window": 5}},
        ]
    }

    base = ProtocolV2(
        data=DataSpec(
            input="data.csv",
            modality="raman",
            label="target",
            metadata_map={
                "sample_id": "id",
                "modality": "modality",
                "label": "target",
            },
        ),
        task=TaskSpec(name="classification", objective="maximize accuracy"),
        preprocess=PreprocessSpec(recipe="basic"),
    )

    expanded = base.expand_recipes(recipe_registry)
    assert expanded.preprocess.recipe is None
    assert [s.component for s in expanded.preprocess.steps] == ["normalize", "smooth"]

    expanded.validate(component_registry={"preprocess": {"normalize", "smooth"}})

    bad = expanded.model_copy(
        update={
            "preprocess": PreprocessSpec(
                recipe=None,
                steps=[PreprocessStep(component="unknown", params={})],
            )
        }
    )

    with pytest.raises(ValueError) as err:
        bad.validate(component_registry={"preprocess": {"normalize", "smooth"}})

    assert "Unknown preprocess components" in str(err.value)
    assert "unknown" in str(err.value)
