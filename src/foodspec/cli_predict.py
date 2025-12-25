"""Backward-compatibility shim for `python -m foodspec.cli_predict`.

Delegates to the canonical argparse CLI in `foodspec.cli.predict`.
This shim will be removed in a future major release.
"""

from __future__ import annotations

import sys

from foodspec.cli.predict import main as _main


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
