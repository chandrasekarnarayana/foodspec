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

ComponentRegistry maps protocol strings to concrete classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Type


@dataclass
class ComponentRegistry:
    """Registry for framework components keyed by category and name.

    Categories covered: preprocess, qc, features, model, splitter, plots, reporters.

    Examples
    --------
    Registering and instantiating a component:
        registry = ComponentRegistry()
        registry.register("preprocess", "normalize", Normalize)
        step = registry.create("preprocess", "normalize", method="area")
    """

    categories: Mapping[str, MutableMapping[str, Type[Any]]] = field(
        default_factory=lambda: {
            "preprocess": {},
            "qc": {},
            "features": {},
            "model": {},
            "splitter": {},
            "plots": {},
            "reporters": {},
        }
    )

    def register(self, category: str, name: str, cls: Type[Any]) -> None:
        """Register a concrete class under a category and name.

        Raises
        ------
        ValueError
            If the category is unknown or the name is already registered.
        """

        if category not in self.categories:
            known = ", ".join(sorted(self.categories))
            raise ValueError(f"Unknown category '{category}'. Known categories: {known}.")

        bucket = self.categories[category]
        if name in bucket:
            raise ValueError(
                f"Component '{name}' already registered in category '{category}'."
            )

        bucket[name] = cls

    def available(self, category: str) -> list[str]:
        """List available component names for a category."""

        if category not in self.categories:
            known = ", ".join(sorted(self.categories))
            raise ValueError(f"Unknown category '{category}'. Known categories: {known}.")
        return sorted(self.categories[category])

    def create(self, category: str, name: str, **params: Any) -> Any:
        """Instantiate a component with provided parameters.

        Raises
        ------
        ValueError
            If the component is not registered in the category.
        """

        if category not in self.categories:
            known = ", ".join(sorted(self.categories))
            raise ValueError(f"Unknown category '{category}'. Known categories: {known}.")

        bucket = self.categories[category]
        if name not in bucket:
            available = ", ".join(sorted(bucket)) or "<none>"
            raise ValueError(
                f"Unknown component '{name}' for category '{category}'. "
                f"Available: {available}."
            )

        cls = bucket[name]
        return cls(**params)


__all__ = ["ComponentRegistry"]
