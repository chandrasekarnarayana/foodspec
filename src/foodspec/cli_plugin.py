"""Backward-compatibility shim for `python -m foodspec.cli_plugin`.

Delegates to the canonical argparse CLI in `foodspec.cli.plugin`.
This shim will be removed in a future major release.

Deprecated:
    Use `foodspec.cli` instead. See docs/migration/v1-to-v2.md.
"""

from __future__ import annotations

import sys
import warnings

from foodspec.cli.plugin import main as _main

warnings.warn(
    "foodspec.cli_plugin is deprecated and will be removed in v2.0.0. "
    "Use foodspec.cli instead. "
    "See docs/migration/v1-to-v2.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
