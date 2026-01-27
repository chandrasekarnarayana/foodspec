#!/usr/bin/env python
"""
Fix from __future__ import placement in Python files.

Moves 'from __future__ import' statements to the top of files,
after shebang and encoding declarations but before other code.
"""

import re
import sys
from pathlib import Path


def fix_future_imports(filepath):
    """Fix from __future__ imports in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, IOError):
        return False

    lines = content.split('\n')

    # Find shebang, encoding, and docstrings
    insert_pos = 0
    in_docstring = False
    docstring_char = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip shebang
        if i == 0 and stripped.startswith('#!'):
            insert_pos = i + 1
            continue

        # Skip encoding
        if i <= 1 and ('coding:' in stripped or 'coding=' in stripped):
            insert_pos = i + 1
            continue

        # Handle module-level docstring
        if i == insert_pos or (i == insert_pos + 1 and lines[insert_pos].strip().startswith('# ')):
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(quote) >= 2:
                    # Single-line docstring
                    insert_pos = i + 1
                    continue
                else:
                    # Multi-line docstring
                    in_docstring = True
                    docstring_char = quote
                    continue
            elif in_docstring:
                if docstring_char in stripped:
                    in_docstring = False
                    insert_pos = i + 1
                    continue

        if not in_docstring:
            break

    # Find and extract from __future__ imports
    future_imports = []
    other_lines = []
    found_future = False

    for line in lines:
        if re.match(r'^\s*from __future__ import', line):
            future_imports.append(line)
            found_future = True
        else:
            other_lines.append(line)

    if not found_future:
        return False

    # Reconstruct file with from __future__ at the right place
    new_lines = []
    added_futures = False

    for i, line in enumerate(other_lines):
        if i == insert_pos and not added_futures:
            new_lines.extend(future_imports)
            new_lines.append(line)
            added_futures = True
        else:
            new_lines.append(line)

    # If we haven't added futures yet, add at the end of what we've processed
    if not added_futures:
        new_lines[insert_pos:insert_pos] = future_imports

    new_content = '\n'.join(new_lines)

    # Only write if changed
    if new_content.strip() != content.strip():
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except IOError:
            return False

    return False

def main():
    """Process all Python files."""
    src_dir = Path('/home/cs/FoodSpec/src/foodspec')

    fixed_files = []
    for py_file in src_dir.rglob('*.py'):
        if fix_future_imports(py_file):
            fixed_files.append(str(py_file))

    print(f"Fixed {len(fixed_files)} files")
    for f in sorted(fixed_files):
        print(f"  {f}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
