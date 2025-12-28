#!/usr/bin/env python3
"""
Fix relative link depths in markdown files under docs/methods/**.

Transforms:
- ../methods/<subpath> -> ../<subpath>
- ../(api|reference|workflows|theory|user-guide|help|troubleshooting|protocols)/ -> ../../<group>/

Run this after restructuring docs to correct common depth mistakes.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

GROUPS_UP_TWO = (
    "api",
    "reference",
    "workflows",
    "theory",
    "user-guide",
    "help",
    "troubleshooting",
    "protocols",
)

def fix_text(text: str) -> str:
    # 1) Drop redundant "methods/" in links like ../methods/preprocessing/foo.md -> ../preprocessing/foo.md
    text = re.sub(r"\]\(\.\./methods/", "](../", text)

    # 2) Ensure top-level siblings are referenced with ../../ when coming from methods/*/*
    pattern = re.compile(r"\]\(\.\./(" + "|".join(GROUPS_UP_TWO) + ")/")
    def _uplvl(m: re.Match) -> str:
        return "](../../%s/" % m.group(1)
    text = pattern.sub(_uplvl, text)

    return text

def main() -> None:
    methods_dir = DOCS / "methods"
    if not methods_dir.exists():
        print("No docs/methods directory found; nothing to do.")
        return

    changed = 0
    files = list(methods_dir.rglob("*.md"))
    for fp in files:
        rel = fp.relative_to(DOCS)
        # Only adjust files that are at least two-level deep (methods/*/*.md)
        if len(rel.parts) < 3:
            continue
        old = fp.read_text(encoding="utf-8")
        new = fix_text(old)
        if new != old:
            fp.write_text(new, encoding="utf-8")
            changed += 1
            print(f"Fixed: {rel}")

    print(f"Updated {changed} files under docs/methods.")

if __name__ == "__main__":
    main()
