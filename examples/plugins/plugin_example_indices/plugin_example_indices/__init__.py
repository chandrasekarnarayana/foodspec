"""
Example feature index plugin.
Registers a simple feature index mapping and a small class wrapper.
"""
from __future__ import annotations


class SimpleFeatureIndex:
    def __init__(self, name: str, features: dict[str, float]):
        self.name = name
        self.features = features

    def keys(self):
        return list(self.features.keys())

    def get(self, key: str):
        return self.features.get(key)


DEMO_INDEX = {
    "I_1742": 1742.0,
    "I_1655": 1655.0,
}


def get_plugins():
    return {
        "protocols": [],
        "vendor_loaders": {},
        "harmonization": {},
        "feature_indices": {
            "demo_index": SimpleFeatureIndex("demo_index", DEMO_INDEX),
        },
        "workflows": {},
    }


def plugin_main():
    return get_plugins()
