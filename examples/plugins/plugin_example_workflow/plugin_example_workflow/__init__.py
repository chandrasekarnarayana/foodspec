"""
Example workflow plugin.
Registers a minimal workflow callable.
"""
from __future__ import annotations

from typing import Any, Dict


def demo_workflow(dataset, **kwargs) -> Dict[str, Any]:
    # Minimal no-op workflow returning a tiny report
    return {
        "samples": getattr(dataset, "x", getattr(dataset, "spectra", None)).shape[0],
        "points": getattr(dataset, "wavenumbers", None).shape[0],
        "params": dict(kwargs),
    }


def get_plugins():
    return {
        "protocols": [],
        "vendor_loaders": {},
        "harmonization": {},
        "feature_indices": {},
        "workflows": {"demo_workflow": demo_workflow},
    }


def plugin_main():
    return get_plugins()
