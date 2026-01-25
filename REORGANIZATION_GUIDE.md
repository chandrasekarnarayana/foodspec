# File Structure Reorganization - Quick Start Guide

## Overview

The FoodSpec repository has been audited and a comprehensive reorganization plan has been created. This guide will help you execute the reorganization safely.

## What Was Found

### Critical Issues (Must Fix)
1. **Dual Source Trees**: Both `src/foodspec/` and `foodspec_rewrite/foodspec/` exist
2. **Duplicate Configs**: Two `pyproject.toml` files causing confusion
3. **Scattered Docs**: 40+ phase documents in 4 different locations
4. **Large Outputs**: 25MB of demo outputs committed to repo
5. **Cache Files**: 642 cache files tracked by git
6. **Built Docs**: 27MB `site/` directory in repo

### Expected Benefits
- **73% size reduction**: 75MB → 20MB (excluding .git/)
- **Single source tree**: No more import ambiguity
- **Clean structure**: Logical organization
- **Faster clones**: Smaller repo size
- **Better maintainability**: Clear where things belong

## Quick Start

### 1. Review the Audit
```bash
# Read the full audit report
cat FILE_STRUCTURE_AUDIT.md

# Or open in your editor
code FILE_STRUCTURE_AUDIT.md
```

### 2. Preview Changes (DRY RUN)
```bash
# This will show what would be changed without modifying anything
python scripts/reorganize_structure.py --dry-run
```

**Expected Output:**
```
=== Step 1: Update .gitignore ===
✓ UPDATE: .gitignore

=== Step 2: Clean Ignored Files ===
✓ CLEAN: ignored files (git clean -fdX)

=== Step 3: Archive Phase Documents ===
✓ CREATE DIR: _internal/phase-history/phase-1-8
✓ MOVE: PHASE10_PROMPT17_COMPLETION.md → _internal/phase-history/phase-1-8/
...

=== Reorganization Summary ===
Total actions: 50+
  CREATE DIR: 8
  MOVE: 35
  REMOVE DIR: 1
  UPDATE: 1
  CLEAN: 1

⚠ DRY RUN: No changes were made
Run with --execute to apply changes
```

### 3. Execute Reorganization
```bash
# When ready, execute the changes
python scripts/reorganize_structure.py --execute
```

**You will be prompted:**
```
EXECUTING CHANGES
Are you sure? This will modify files. (yes/no): 
```

Type `yes` to proceed.

### 4. Verify Changes
```bash
# Check what was changed
git status

# Review the diff
git diff --stat

# Run tests to ensure nothing broke
pytest tests/ -v

# Check a few examples still work
python examples/quickstarts/oil_authentication_quickstart.py
```

### 5. Commit the Reorganization
```bash
# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "refactor: Reorganize repository file structure

- Remove foodspec_rewrite/ directory (merge complete)
- Archive 40+ phase documents in _internal/phase-history/
- Reorganize examples/ into logical subdirectories
- Reorganize scripts/ into functional categories
- Update .gitignore to exclude build/demo outputs
- Clean 642 cache files and 27MB built docs

Impact: 73% size reduction, eliminates import ambiguity,
improves discoverability and maintainability.

See: FILE_STRUCTURE_AUDIT.md for full details"

# Push to remote
git push origin main
```

## What Gets Changed

### Files/Directories Moved

**Phase Documents** → `_internal/phase-history/`:
- 9 root-level `PHASE*.md` files
- 2 JOSS documents
- 17 foodspec_rewrite/ docs

**Examples** → Subdirectories:
- `examples/quickstarts/` (6 files)
- `examples/advanced/` (8 files)
- `examples/validation/` (3 files)
- `examples/new-features/` (8 files)

**Scripts** → Subdirectories:
- `scripts/development/` (3 files)
- `scripts/documentation/` (5 files)
- `scripts/maintenance/` (4 files)
- `scripts/workflows/` (1 file)

### Files/Directories Removed

- `foodspec_rewrite/` (entire directory)
- `site/` (via .gitignore)
- `outputs/` (via .gitignore)
- `demo_*/` (via .gitignore)
- `*_runs/`, `*_output/`, `*_export/` (via .gitignore)
- All `__pycache__/` and `.pytest_cache/` (via .gitignore)

### Files Created/Updated

- `.gitignore` (updated with comprehensive rules)
- `_internal/phase-history/README.md` (new index file)

## Rollback Plan

If something goes wrong:

### Before Committing
```bash
# Reset all changes
git reset --hard HEAD

# Clean untracked files
git clean -fd
```

### After Committing (But Before Pushing)
```bash
# Undo the last commit but keep changes
git reset --soft HEAD~1

# Or completely remove the commit
git reset --hard HEAD~1
```

### After Pushing
```bash
# Revert the commit
git revert <commit-sha>
git push origin main
```

## Troubleshooting

### "git mv" Fails
If you see errors about files not being tracked:
```bash
# The script will automatically fall back to regular mv
# But you can manually track the file:
git add <file>
```

### Import Errors After Reorganization
```bash
# Verify your Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall in development mode
pip install -e .

# Clear any cached imports
find . -name "__pycache__" -exec rm -rf {} +
```

### Tests Fail
```bash
# Check which tests are failing
pytest tests/ -v --tb=short

# Most common issue: relative imports in examples
# Update sys.path in affected files
```

## Timeline

Estimated time: **30-60 minutes**

- Step 1-2 (gitignore & clean): 5 min
- Step 3-5 (archive & remove): 10 min
- Step 6-7 (reorganize): 10 min
- Step 8 (verify tests): 20 min
- Step 9 (commit & push): 5 min
- Buffer: 10 min

## Safety Features

The reorganization script includes:

1. **Dry-run mode**: Preview all changes first
2. **Git operations**: Uses `git mv` to preserve history
3. **Confirmation prompt**: Asks before making changes
4. **Error handling**: Catches and reports errors gracefully
5. **Action logging**: Records every operation performed

## Support

If you encounter issues:

1. Check the audit report: [FILE_STRUCTURE_AUDIT.md](FILE_STRUCTURE_AUDIT.md)
2. Review the script: [scripts/reorganize_structure.py](scripts/reorganize_structure.py)
3. Open an issue on GitHub
4. Contact the maintainer

## After Reorganization

### Update Documentation
- Update `CONTRIBUTING.md` with new paths
- Update any README references to old paths
- Update CI/CD config if it references old paths

### Communicate Changes
- Announce in GitHub Discussions
- Update any external documentation
- Notify regular contributors

### Monitor
- Watch for issues related to new structure
- Update paths in any automation scripts
- Fix any remaining hardcoded paths

---

**Status**: Ready to execute  
**Risk Level**: Low (reversible changes)  
**Recommended**: Yes  

Execute when you have 30-60 minutes to complete and verify the reorganization.
