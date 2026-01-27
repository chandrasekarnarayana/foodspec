from __future__ import annotations

"""
Registry for vendor loader plugins.

Plugins can register callables that ingest a vendor-specific path and return either
an IngestResult, a FoodSpectrumSet, or a SpectralDataset-like object.
"""


from pathlib import Path
from typing import Callable, Dict, Optional

VendorLoader = Callable[..., object]


class VendorLoaderRegistry:
    def __init__(self) -> None:
        self._loaders: Dict[str, VendorLoader] = {}

    def register(self, name: str, loader: VendorLoader, *, override: bool = False) -> None:
        key = name.lower()
        if not override and key in self._loaders:
            raise ValueError(f"Vendor loader '{name}' already registered")
        self._loaders[key] = loader

    def get(self, name: str) -> Optional[VendorLoader]:
        return self._loaders.get(name.lower())

    def unregister(self, name: str) -> None:
        self._loaders.pop(name.lower(), None)

    def available(self):
        return sorted(self._loaders)

    def clear(self) -> None:
        self._loaders.clear()

    def load(self, name: str, path: str | Path, **kwargs):
        loader = self.get(name)
        if loader is None:
            raise KeyError(f"Vendor loader '{name}' not found; available: {self.available()}")
        return loader(path, **kwargs)


vendor_loader_registry = VendorLoaderRegistry()
