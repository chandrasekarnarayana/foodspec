"""
Feature registry integration with ProtocolV2 and feature extractors.
"""

import numpy as np

from foodspec.core.protocol import DataSpec, FeatureSpec, ProtocolV2, TaskSpec
from foodspec.core.registry import ComponentRegistry, register_default_feature_components


def _make_protocol(modules: list[str]) -> ProtocolV2:
    return ProtocolV2(
        data=DataSpec(
            input="dummy.csv",
            modality="raman",
            label="target",
            metadata_map={
                "sample_id": "sample_id",
                "modality": "modality",
                "label": "target",
            },
        ),
        task=TaskSpec(name="demo", objective="classification", constraints={}),
        features=FeatureSpec(strategy="manual", modules=modules),
    )


def test_protocol_accepts_registered_feature_modules() -> None:
    registry = ComponentRegistry()
    register_default_feature_components(registry)

    proto = _make_protocol(["pca", "feature_union", "stability_selection_selector"])

    proto.validate(component_registry={"features": registry.available("features")})


def test_registry_creates_and_runs_feature_extractors() -> None:
    registry = ComponentRegistry()
    register_default_feature_components(registry)

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 12))
    X_test = rng.normal(size=(5, 12))
    y = rng.integers(0, 2, size=30)

    feature_union = registry.create(
        "features",
        "feature_union",
        extractors=[
            registry.create("features", "pca", n_components=3, seed=0),
            registry.create("features", "pls", n_components=2),
        ],
    )

    feature_union.fit(X_train, y)
    feature_set = feature_union.transform(X_test)

    assert feature_set.Xf.shape == (X_test.shape[0], 5)
    assert len(feature_set.feature_names) == 5
    assert "stability_selection_selector" in registry.available("features")