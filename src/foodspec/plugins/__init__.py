"""
Plugin bootstrap helpers.

`load_plugins()` discovers entry points under `foodspec.plugins` and
registers vendor loaders, feature indices, and workflows into runtime
registries for use across the library.
"""

from __future__ import annotations

from foodspec.plugin import PluginManager
from foodspec.plugins.indices import feature_index_registry
from foodspec.plugins.loaders import vendor_loader_registry
from foodspec.plugins.workflows import workflow_registry

_plugins_loaded = False
_cached_manager: PluginManager | None = None


def load_plugins(force: bool = False) -> PluginManager:
    """Discover plugins and populate runtime registries."""
    global _plugins_loaded, _cached_manager
    if _plugins_loaded and not force and _cached_manager is not None:
        return _cached_manager

    pm = PluginManager().discover()
    if force:
        vendor_loader_registry.clear()
        feature_index_registry.clear()
        workflow_registry.clear()

    for name, entry in pm.vendor_loaders.items():
        vendor_loader_registry.register(name, entry.obj, override=True)
    for name, entry in pm.feature_indices.items():
        feature_index_registry.register(name, entry.obj, override=True)
    for name, entry in pm.workflows.items():
        workflow_registry.register(name, entry.obj, override=True)

    _plugins_loaded = True
    _cached_manager = pm
    return pm


def ensure_plugins_loaded() -> PluginManager:
    """Idempotent wrapper for `load_plugins` to avoid repeated discovery."""
    return load_plugins(force=False)
