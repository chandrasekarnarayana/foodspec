# Engineering Rules

These rules ensure the codebase stays maintainable and reproducible.

## Core rules
1. Mindmap modules are the primary implementation.
2. Old import paths are kept via deprecation shims for one version.
3. All new public APIs include type hints and docstrings.
4. Runs must write logs, a manifest, and a JSON summary.
5. Tests cover new modules and CLI surfaces.

