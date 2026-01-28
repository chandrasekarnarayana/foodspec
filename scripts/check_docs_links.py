#!/usr/bin/env python3
"""Simple docs link checker for markdown files."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def collect_markdown_files(docs_root: Path) -> list[Path]:
    return list(docs_root.rglob("*.md"))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    docs_root = repo_root / "docs"
    if not docs_root.exists():
        print("docs/ not found")
        return 1

    link_re = re.compile(r"\]\(([^)]+)\)")
    broken = []

    for path in collect_markdown_files(docs_root):
        text = path.read_text(errors="ignore")
        for match in link_re.findall(text):
            link = match.strip()
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            if link.startswith("#"):
                continue
            link = link.split("#")[0]
            if not link:
                continue
            target = (path.parent / link).resolve()
            if target.exists():
                continue
            repo_target = (repo_root / link).resolve()
            if repo_target.exists():
                continue
            broken.append((path, link))

    if broken:
        print("Broken links:")
        for path, link in broken:
            rel = path.relative_to(repo_root)
            print(f"- {rel}: {link}")
        return 1

    print("No broken links found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
