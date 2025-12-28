#!/usr/bin/env python3
"""Fix all subfolder link depths (tutorials and workflows)."""

import re
from pathlib import Path

# Subfolders that need ../../ instead of ../
SUBFOLDERS = [
    'tutorials/beginner',
    'tutorials/intermediate',
    'tutorials/advanced',
]

# Patterns to fix: ../ -> ../../ for these targets
TARGETS = [
    'methods/',
    'reference/',
    'api/',
    'theory/',
    'user-guide/',
    'protocols/',
    'troubleshooting/',
    'workflows/'
]

def fix_file(file_path: Path) -> int:
    """Fix link depths in a subfolder file."""
    content = file_path.read_text()
    original = content
    changes = 0
    
    for target in TARGETS:
        # Match links like ../methods/ and replace with ../../methods/
        pattern = rf'\(\.\.\/{re.escape(target)}'
        replacement = f'(../../{target}'
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            matches = len(re.findall(pattern, content))
            changes += matches
            content = new_content
            print(f"  Fixed {matches} ../{target} → ../../{target}")
    
    if content != original:
        file_path.write_text(content)
        return changes
    return 0

def main():
    docs_dir = Path('docs')
    total_changes = 0
    
    for subfolder in SUBFOLDERS:
        folder_path = docs_dir / subfolder
        if not folder_path.exists():
            continue
        
        for md_file in folder_path.glob('*.md'):
            print(f"\nProcessing {md_file.relative_to(docs_dir)}...")
            changes = fix_file(md_file)
            if changes > 0:
                total_changes += changes
                print(f"  ✓ Total: {changes} links updated")
    
    print(f"\n{'='*70}")
    print(f"Fixed {total_changes} subfolder link depths")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
