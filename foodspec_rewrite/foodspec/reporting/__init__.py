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
Reporting module: Report templates, export formats, formatting utilities.

Generating reports of results:
    from foodspec.reporting import generate_html_report
    report_path = generate_html_report(...)
"""

from foodspec.reporting.report import DEFAULT_TEMPLATE, generate_html_report

__all__ = ["generate_html_report", "DEFAULT_TEMPLATE"]
