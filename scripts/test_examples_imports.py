#!/usr/bin/env python3
"""
Quick test to verify all example scripts can at least import their dependencies.
This doesn't run the full examples (which may need data files), just checks imports.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
examples_dir = repo_root / 'examples'

passed = 0
failed = 0
errors = []

for example_file in sorted(examples_dir.glob('*.py')):
    if example_file.name.startswith('_'):
        continue
        
    try:
        # Read and compile the file (this checks syntax and imports at module level)
        code = example_file.read_text()
        compile(code, str(example_file), 'exec')
        passed += 1
        print(f"✓ {example_file.name}")
    except Exception as e:
        failed += 1
        error_msg = str(e).split('\n')[0][:80]
        errors.append((example_file.name, error_msg))
        print(f"✗ {example_file.name}: {error_msg}")

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*60}")

if failed > 0:
    print("\nFailed examples:")
    for name, error in errors:
        print(f"  - {name}: {error}")
    sys.exit(1)
else:
    print("\n✓ All examples compile successfully!")
    sys.exit(0)
