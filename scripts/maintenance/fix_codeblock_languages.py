#!/usr/bin/env python3
"""
Auto-tag unlabeled code fences in Markdown files under docs/ with language hints.
Heuristics:
- python: contains 'import ', 'from ', 'def ', 'print(', 'plt.'
- bash: lines start with common CLI commands (python, pip, mkdocs, foodspec, git, cd, ls, sed, awk, echo) or include '&&', '; \\'
- yaml: first non-empty line ends with ':' or typical YAML keys/structure
- json: starts with '{' and ends with '}'
- toml: contains '=' assignments with section headers '[...]'
- mermaid: contains 'flowchart' or 'graph' keywords (skip if already labeled)
- default: 'plaintext'

Only modifies fences like ``` with no language. Preserves indentation.
"""

import re
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"

fence_open_re = re.compile(r"^(?P<indent>\s*)```\s*$")
fence_open_labeled_re = re.compile(r"^(?P<indent>\s*)```\s*(\w+)\s*$")
fence_close_re = re.compile(r"^(?P<indent>\s*)```\s*$")
fence_tilde_open_re = re.compile(r"^(?P<indent>\s*)~~~\s*$")
fence_tilde_open_labeled_re = re.compile(r"^(?P<indent>\s*)~~~\s*(\w+)\s*$")
fence_tilde_close_re = re.compile(r"^(?P<indent>\s*)~~~\s*$")

BASH_PREFIXES = (
    "python ",
    "pip ",
    "mkdocs ",
    "foodspec ",
    "git ",
    "cd ",
    "ls ",
    "pwd ",
    "sed ",
    "awk ",
    "echo ",
    "rm ",
    "cp ",
    "mv ",
    "make ",
    "pytest ",
    "npm ",
)


def detect_lang(block_text: str) -> str:
    text = block_text.strip()
    lines = [line.rstrip() for line in block_text.splitlines()]
    lower_text = text.lower()

    # Mermaid
    if "flowchart" in lower_text or lower_text.startswith("graph"):
        return "mermaid"

    # JSON
    if text.startswith("{") and text.endswith("}"):
        return "json"

    # YAML (simple heuristic: many lines with key: value)
    yaml_like = 0
    for line in lines:
        ls = line.strip()
        if not ls:
            continue
        if ls.startswith("#"):
            yaml_like += 1
            continue
        if ":" in ls and not ls.startswith(("http", "https")):
            yaml_like += 1
    if yaml_like >= max(2, len(lines) // 4):
        return "yaml"

    # TOML
    if any(line.strip().startswith("[") and line.strip().endswith("]") for line in lines) and any(
        "=" in line and not line.strip().startswith("#") for line in lines
    ):
        return "toml"

    # Python
    if any(kw in lower_text for kw in ("\nimport ", "\nfrom ", "def ", "print(", "plt.", "yaml.safe_load", "sklearn.")):
        return "python"

    # Bash/CLI
    bash_score = 0
    for line in lines:
        ls = line.strip()
        if any(ls.startswith(p) for p in BASH_PREFIXES):
            bash_score += 2
        if "&&" in ls or ls.endswith(" \\") or ls.startswith("#!/bin/bash"):
            bash_score += 1
    if bash_score >= 2:
        return "bash"

    return "plaintext"


def process_file(md_path: Path) -> bool:
    original = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = original.splitlines()
    out_lines = []
    i = 0
    changed = False

    in_fence = False
    fence_type = None  # 'backtick' or 'tilde'

    while i < len(lines):
        line = lines[i]

        # Helper matchers for current line
        m_bt_labeled = fence_open_labeled_re.match(line)
        m_bt_unlabeled = fence_open_re.match(line)
        m_bt_close = fence_close_re.match(line)
        m_tl_labeled = fence_tilde_open_labeled_re.match(line)
        m_tl_unlabeled = fence_tilde_open_re.match(line)
        m_tl_close = fence_tilde_close_re.match(line)

        if not in_fence:
            # Labeled fence opens (backtick or tilde)
            if m_bt_labeled:
                in_fence = True
                fence_type = "backtick"
                out_lines.append(line)
                i += 1
                continue
            if m_tl_labeled:
                in_fence = True
                fence_type = "tilde"
                out_lines.append(line)
                i += 1
                continue

            # Unlabeled fence open: capture block, detect lang, rewrite opening
            if m_bt_unlabeled:
                indent = m_bt_unlabeled.group("indent")
                j = i + 1
                block = []
                while j < len(lines):
                    if fence_close_re.match(lines[j]):
                        break
                    block.append(lines[j])
                    j += 1
                if j >= len(lines):
                    # No closing fence; copy as-is
                    out_lines.append(line)
                    i += 1
                    continue
                lang = detect_lang("\n".join(block))
                out_lines.append(f"{indent}```{lang}")
                out_lines.extend(block)
                out_lines.append(lines[j])  # closing fence
                changed = True
                i = j + 1
                continue

            if m_tl_unlabeled:
                indent = m_tl_unlabeled.group("indent")
                j = i + 1
                block = []
                while j < len(lines):
                    if fence_tilde_close_re.match(lines[j]):
                        break
                    block.append(lines[j])
                    j += 1
                if j >= len(lines):
                    out_lines.append(line)
                    i += 1
                    continue
                lang = detect_lang("\n".join(block))
                out_lines.append(f"{indent}~~~{lang}")
                out_lines.extend(block)
                out_lines.append(lines[j])
                changed = True
                i = j + 1
                continue

            # Not a fence, just copy
            out_lines.append(line)
            i += 1
            continue

        else:
            # We are inside a fence; handle proper closings
            if fence_type == "backtick":
                if m_bt_close:
                    out_lines.append(line)
                    in_fence = False
                    fence_type = None
                    i += 1
                    continue
                # If a labeled open appears inside a fence, it's likely a mis-labeled closing from a prior pass; fix to closing
                if m_bt_labeled:
                    out_lines.append(m_bt_labeled.group("indent") + "```")
                    in_fence = False
                    fence_type = None
                    changed = True
                    i += 1
                    continue
            elif fence_type == "tilde":
                if m_tl_close:
                    out_lines.append(line)
                    in_fence = False
                    fence_type = None
                    i += 1
                    continue
                if m_tl_labeled:
                    out_lines.append(m_tl_labeled.group("indent") + "~~~")
                    in_fence = False
                    fence_type = None
                    changed = True
                    i += 1
                    continue

            # Regular content inside fence
            out_lines.append(line)
            i += 1

    if changed:
        md_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed


def main():
    md_files = sorted(DOCS_ROOT.rglob("*.md"))
    total = 0
    changed = 0
    for md in md_files:
        total += 1
        try:
            if process_file(md):
                changed += 1
        except Exception as e:
            print(f"WARN: Failed to process {md}: {e}")
    print(f"Processed {total} files; updated {changed} with language tags.")


if __name__ == "__main__":
    main()
