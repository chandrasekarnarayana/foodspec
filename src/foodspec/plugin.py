"""
Plugin system for FoodSpec.

Plugins can register:
- protocol templates
- vendor loaders
- harmonization strategies
- feature indices
- workflows

Discovery via entry points: `foodspec.plugins`.
Each entry point should expose a `get_plugins()` function returning a dict with keys:
    {"protocols": [...], "vendor_loaders": {...}, "harmonization": {...}, "feature_indices": {...}, "workflows": {...}}
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PluginEntry:
    name: str
    module: str
    obj: object


@dataclass
class PluginManager:
    protocols: List[PluginEntry] = field(default_factory=list)
    vendor_loaders: Dict[str, PluginEntry] = field(default_factory=dict)
    harmonization: Dict[str, PluginEntry] = field(default_factory=dict)
    feature_indices: Dict[str, PluginEntry] = field(default_factory=dict)
    workflows: Dict[str, PluginEntry] = field(default_factory=dict)

    def discover(self) -> "PluginManager":
        try:
            from importlib.metadata import entry_points  # Py>=3.10
        except ImportError:
            from importlib_metadata import entry_points  # type: ignore

        eps = entry_points()
        group = eps.select(group="foodspec.plugins") if hasattr(eps, "select") else eps.get("foodspec.plugins", [])
        for ep in group:
            try:
                mod = ep.load()
            except (ModuleNotFoundError, ImportError, AttributeError):
                # Skip plugins with missing modules (e.g., test fixtures)
                continue
            try:
                payload = mod.get_plugins()
            except Exception:
                continue
            for p in payload.get("protocols", []):
                self.protocols.append(PluginEntry(name=getattr(p, "name", str(p)), module=ep.module, obj=p))
            for k, v in payload.get("vendor_loaders", {}).items():
                self.vendor_loaders[k] = PluginEntry(name=k, module=ep.module, obj=v)
            for k, v in payload.get("harmonization", {}).items():
                self.harmonization[k] = PluginEntry(name=k, module=ep.module, obj=v)
            for k, v in payload.get("feature_indices", {}).items():
                self.feature_indices[k] = PluginEntry(name=k, module=ep.module, obj=v)
            for k, v in payload.get("workflows", {}).items():
                self.workflows[k] = PluginEntry(name=k, module=ep.module, obj=v)
        return self


def install_plugin(module_path: str):
    """
    Placeholder: in a real implementation, we might pip-install from a source.
    Here we simply attempt to import to verify availability.
    """
    importlib.import_module(module_path)
    return True
