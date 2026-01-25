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
CLI module: Command-line interface commands and utilities.

Running FoodSpec from the command line:
    foodspec run --protocol config.yaml --outdir runs/exp1 --seed 42
    foodspec predict --bundle ./bundle --input new_data.csv --outdir predictions
    foodspec report --run-dir runs/exp1 --output report.html
"""

from .main import app, main

__all__ = ["app", "main"]
