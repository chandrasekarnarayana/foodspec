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

Preprocessing recipes for Raman and FTIR workflows.
"""

from __future__ import annotations

from typing import Dict, List, Mapping


def raman_auth_default() -> List[Dict]:
    """Default Raman authentication recipe (despike -> baseline -> SNV)."""

    return [
        {"component": "DespikeHampel", "params": {"window_size": 3, "threshold": 3.0}},
        {"component": "AsLSBaseline", "params": {"lam": 1e5, "p": 0.01, "n_iter": 10}},
        {"component": "SNV", "params": {}},
    ]


def ftir_auth_default() -> List[Dict]:
    """Default FTIR authentication recipe (MSC -> vector normalize)."""

    return [
        {"component": "MultiplicativeScatterCorrection", "params": {}},
        {"component": "VectorNormalize", "params": {}},
    ]


_BUILTINS: Mapping[str, List[Dict]] = {
    "raman_auth_default": raman_auth_default(),
    "ftir_auth_default": ftir_auth_default(),
}


def resolve_recipe(name: str) -> List[Dict]:
    """Return recipe steps for a named recipe or raise with available names."""

    if name not in _BUILTINS:
        available = ", ".join(sorted(_BUILTINS))
        raise ValueError(f"Unknown recipe '{name}'. Available: {available}.")
    return _BUILTINS[name]


__all__ = ["resolve_recipe", "raman_auth_default", "ftir_auth_default"]
