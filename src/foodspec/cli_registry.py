from __future__ import annotations

"""Backward-compatibility shim for `python -m foodspec.cli_registry`.

Delegates to the canonical argparse CLI in `foodspec.cli.registry`.
This shim will be removed in a future major release.
"""

"""
cli_registry - DEPRECATED

.. deprecated:: 1.1.0
    This module is deprecated and will be removed in v2.0.0.
    Use foodspec.cli instead.

This module is maintained for backward compatibility only.
All new code should use the modern API.

Migration Guide:
    Old: from foodspec.cli_registry import ...
    New: from foodspec.cli import ...

See: docs/migration/v1-to-v2.md
"""

import warnings

warnings.warn(
    "foodspec.cli_registry is deprecated and will be removed in v2.0.0. "
    "Use foodspec.cli instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Original module content continues below...
# ==============================================





import sys

from foodspec.cli.registry import main as _main


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
