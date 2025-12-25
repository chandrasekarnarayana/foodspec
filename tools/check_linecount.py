#!/usr/bin/env python3
"""
Check that no Python source file exceeds a maximum line count.

This script enforces a guideline that production code modules should be
concise and focused. Test files are excluded.

Usage:
    python tools/check_linecount.py [--max-lines 600] [--exclude-tests]

Exit:
    0 if all files pass
    1 if any file exceeds the limit
"""

import sys
from pathlib import Path


def check_linecount(
    src_dir: str = "src",
    max_lines: int = 600,
    exclude_tests: bool = True,
) -> int:
    """
    Check Python files in src_dir for line count.

    Args:
        src_dir: Root directory to scan for .py files
        max_lines: Maximum allowed lines per file
        exclude_tests: If True, skip test files

    Returns:
        0 if all pass, 1 if any fail
    """
    src_path = Path(src_dir)
    violations = []

    for py_file in sorted(src_path.rglob("*.py")):
        if exclude_tests and "test_" in py_file.name:
            continue

        line_count = len(py_file.read_text(encoding="utf-8").splitlines())

        if line_count > max_lines:
            violations.append((str(py_file.relative_to(Path.cwd())), line_count))

    if violations:
        print(f"❌ Found {len(violations)} file(s) exceeding {max_lines} lines:")
        for filepath, count in violations:
            print(f"  {filepath}: {count} lines")
        return 1

    print(f"✓ All files in {src_dir}/ are under {max_lines} lines")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check Python source files do not exceed max line count."
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=600,
        help="Maximum allowed lines per file (default: 600)",
    )
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        default=True,
        help="Exclude test files (default: True)",
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        default="src",
        help="Source directory to check (default: src)",
    )

    args = parser.parse_args()
    sys.exit(
        check_linecount(
            src_dir=args.src_dir,
            max_lines=args.max_lines,
            exclude_tests=args.exclude_tests,
        )
    )
