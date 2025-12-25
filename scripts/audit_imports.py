#!/usr/bin/env python3
"""Audit all imports in docs, examples, and notebooks to ensure they reference actual package code."""

import re
import sys
from collections import defaultdict
from pathlib import Path


# Find all Python imports in markdown and Python files
def extract_imports_from_md(content: str) -> list[str]:
    """Extract Python imports from code blocks in markdown."""
    imports = []
    # Match Python code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    for block in code_blocks:
        # Handle multiline imports
        block_lines = block.split('\n')
        current_import = []
        in_multiline = False

        for line in block_lines:
            stripped = line.strip()

            # Start of import
            if stripped.startswith(('from foodspec', 'import foodspec')):
                current_import = [line]
                # Check if multiline
                if '(' in stripped and ')' not in stripped:
                    in_multiline = True
                else:
                    imports.append(line.strip())
                    current_import = []
            # Continuation of multiline import
            elif in_multiline:
                current_import.append(line)
                if ')' in stripped:
                    # End of multiline, join and add
                    full_import = '\n'.join(current_import)
                    imports.append(full_import)
                    current_import = []
                    in_multiline = False

    return imports

def extract_imports_from_py(content: str) -> list[str]:
    """Extract foodspec imports from Python files."""
    imports = []
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(('from foodspec', 'import foodspec')):
            # Remove comments
            line = line.split('#')[0].strip()
            if line:
                # Check if multiline
                if '(' in line and ')' not in line:
                    # Collect until closing paren
                    full_import = [lines[i]]
                    i += 1
                    while i < len(lines) and ')' not in lines[i]:
                        full_import.append(lines[i])
                        i += 1
                    if i < len(lines):
                        full_import.append(lines[i])
                    imports.append('\n'.join(full_import))
                else:
                    imports.append(line)
        i += 1
    return imports

def test_import(import_stmt: str) -> tuple[bool, str]:
    """Test if an import statement works."""
    try:
        exec(import_stmt)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def main():
    repo_root = Path(__file__).parent.parent

    # Scan locations
    locations = {
        'docs': repo_root / 'docs',
        'examples': repo_root / 'examples',
    }

    all_imports = defaultdict(list)
    failed_imports = defaultdict(list)

    # Scan markdown files
    for md_file in locations['docs'].rglob('*.md'):
        if 'archive' in str(md_file):
            continue
        content = md_file.read_text()
        imports = extract_imports_from_md(content)
        for imp in imports:
            all_imports[imp].append(str(md_file.relative_to(repo_root)))

    # Scan Python files
    for py_file in locations['examples'].rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        content = py_file.read_text()
        imports = extract_imports_from_py(content)
        for imp in imports:
            all_imports[imp].append(str(py_file.relative_to(repo_root)))

    # Test all unique imports
    print("=" * 80)
    print("IMPORT AUDIT REPORT")
    print("=" * 80)
    print()

    unique_imports = sorted(all_imports.keys())
    total = len(unique_imports)
    passed = 0
    failed = 0

    print(f"Testing {total} unique import statements...\n")

    for imp in unique_imports:
        success, error = test_import(imp)
        if success:
            passed += 1
        else:
            failed += 1
            failed_imports[imp] = (error, all_imports[imp])
            print(f"✗ FAILED: {imp}")
            print(f"  Error: {error}")
            print(f"  Used in: {', '.join(all_imports[imp][:3])}")
            if len(all_imports[imp]) > 3:
                print(f"           ... and {len(all_imports[imp]) - 3} more files")
            print()

    print("=" * 80)
    print(f"SUMMARY: {passed}/{total} imports work ({passed*100//total}%)")
    print(f"         {failed} imports FAILED")
    print("=" * 80)

    if failed > 0:
        print("\nFailed imports need to be fixed:")
        for imp, (error, files) in failed_imports.items():
            print(f"  - {imp}")
        sys.exit(1)
    else:
        print("\n✓ All imports are valid!")
        sys.exit(0)

if __name__ == '__main__':
    main()
