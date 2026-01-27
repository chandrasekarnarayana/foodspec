#!/usr/bin/env python3
"""Docs validation helper for CI.

Supports:
  --full       : build docs (strict) + link check
  --strict     : build docs (strict) + link check
  --skip-build : only run markdown link check
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FoodSpec docs.")
    parser.add_argument("--full", action="store_true", help="Run full build + link checks.")
    parser.add_argument("--strict", action="store_true", help="Build docs in strict mode + link checks.")
    parser.add_argument("--skip-build", action="store_true", help="Skip mkdocs build; only check markdown links.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        print("docs/ not found")
        return 1

    if not args.skip_build:
        mkdocs_cmd = ["mkdocs", "build", "--clean"]
        if args.full or args.strict:
            mkdocs_cmd.append("--strict")
        mkdocs_cmd.extend(["--site-dir", "site"])
        code = _run(mkdocs_cmd)
        if code != 0:
            return code

    # Markdown link check (source docs)
    check_cmd = [sys.executable, str(repo_root / "scripts" / "check_docs_links.py")]
    return _run(check_cmd)


if __name__ == "__main__":
    sys.exit(main())
