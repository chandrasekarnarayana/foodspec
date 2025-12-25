"""Backward-compatibility shim for `python -m foodspec.cli_protocol`.

Delegates to the canonical argparse CLI in `foodspec.cli.protocol`.
This shim will be removed in a future major release.
"""
from __future__ import annotations

import sys

from foodspec.cli.protocol import main as _main


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
