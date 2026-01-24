# Safe Refactoring Workflow: Git Branch Strategy

**Status**: Recommended approach for FoodSpec refactoring  
**Created**: 2026-01-24  
**Approach**: Option A (New Branch + Delete in Branch)

---

## Recommended Approach: Option A

**Best Practice for FoodSpec Refactor**

This approach maintains full git history while allowing a clean refactor:

### Why Option A?

✅ **Git history preserved** — Old code always recoverable  
✅ **Clean separation** — New code isolated from old  
✅ **Code review friendly** — Easy to see before/after  
✅ **Rollback safe** — Can revert branch if needed  
✅ **Merge friendly** — Clear merge strategy to main

---

## Workflow: Step-by-Step

### Phase 0: Preparation (Current — Main Branch)

```bash
# Current state: main branch
# ✅ Created: Engineering rules & compatibility plan
# ✅ Created: Public API inventory & migration strategy
# ✅ CONTRIBUTING.md updated with 7 rules

git status
# Should show no uncommitted changes (or staged for next commit)
```

### Phase 1: Create Refactor Branch

```bash
# Create feature branch for Phase 1 (protocol-driven core)
git checkout -b phase-1/protocol-driven-core

# Confirm you're on the new branch
git branch
# * phase-1/protocol-driven-core
#   main
```

### Phase 2: Build New Structure in `src/foodspec/core/`

**Inside the branch, create the new architecture:**

```
src/foodspec/
├── __init__.py                    # Re-exports (stable interface)
├── core/                          # ✨ NEW (Phase 1)
│   ├── __init__.py                # Unified API (FoodSpec class)
│   ├── api.py                     # Main FoodSpec entry point
│   ├── spectrum.py                # Spectrum class
│   ├── run_record.py              # Metadata/provenance
│   ├── output_bundle.py           # Results container
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── baseline.py            # ← Moved from spectral_dataset.py
│   │   ├── normalization.py
│   │   └── smoothing.py
│   ├── dataset.py                 # ← Refactored FoodSpectrumSet
│   ├── hyperspectral.py           # ← Refactored HyperSpectralCube
│   └── ...
│
├── spectral_dataset.py            # ✅ LEGACY (re-exports with compat warnings)
├── io.py                          # ✅ Stays here OR move to core/io/
├── stats.py                       # ✅ Stays here OR move to core/stats/
└── ... (other existing modules)
```

### Phase 3: Maintain Backward Compatibility

**Create re-export wrappers in old locations:**

```python
# src/foodspec/spectral_dataset.py (LEGACY, kept for compat)

"""
DEPRECATED: Functions moved to foodspec.core.preprocessing.

This module is maintained for backward compatibility only.
See docs/developer-guide/COMPATIBILITY_PLAN.md
"""

import warnings

def baseline_als(*args, **kwargs):
    """.. deprecated:: 1.1.0
    Use foodspec.core.preprocessing.baseline.baseline_als instead.
    """
    warnings.warn(
        "baseline_als from foodspec.spectral_dataset is deprecated. "
        "Use foodspec.core.preprocessing.baseline.baseline_als instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from foodspec.core.preprocessing.baseline import baseline_als as _impl
    return _impl(*args, **kwargs)

# ... re-export other functions
```

### Phase 4: Update Top-Level `__init__.py`

**In `src/foodspec/__init__.py`, import from new locations:**

```python
"""foodspec: Protocol-driven spectroscopy framework."""

__version__ = "1.1.0"

# ========== Import from NEW locations ==========
from foodspec.core.api import FoodSpec
from foodspec.core.spectrum import Spectrum
from foodspec.core.preprocessing.baseline import baseline_als
# ... more new imports

# ========== Re-export (public API remains stable) ==========
__all__ = [
    "__version__",
    "FoodSpec",
    "Spectrum",
    "baseline_als",
    # ... all public APIs
]
```

### Phase 5: Testing & Verification

```bash
# Still in phase-1/protocol-driven-core branch

# Run all tests (should still pass via re-exports)
pytest tests/ -v

# Check for deprecation warnings
pytest tests/ -W error::DeprecationWarning --ignore=tests/test_backward_compat.py

# Check coverage
pytest tests/ --cov=src/foodspec --cov-fail-under=80

# Type checking
mypy src/ --strict

# Linting
ruff check src/ --fix
ruff format src/
```

### Phase 6: Backward Compatibility Tests

```python
# tests/test_backward_compat.py

import pytest
import warnings

class TestBackwardCompatibility:
    def test_old_import_works(self):
        """Old import path still works."""
        with pytest.warns(DeprecationWarning):
            from foodspec.spectral_dataset import baseline_als
            assert callable(baseline_als)
    
    def test_new_import_works(self):
        """New import path works without warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            from foodspec.core.preprocessing.baseline import baseline_als
            assert callable(baseline_als)
    
    def test_top_level_import_works(self):
        """Top-level import always works."""
        from foodspec import baseline_als
        assert callable(baseline_als)
```

### Phase 7: Commit on Branch

```bash
# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "Phase 1: Protocol-driven core architecture

- Created foodspec.core module with FoodSpec unified API
- Moved baseline functions to core.preprocessing
- Added backward compatibility re-exports
- Maintained public API surface (no breaking changes in v1.1.0)
- All tests passing (coverage >= 80%)

See docs/developer-guide/COMPATIBILITY_PLAN.md for details."

# View commits on this branch
git log --oneline -10
```

### Phase 8: Push Branch for Review

```bash
# Push to GitHub
git push -u origin phase-1/protocol-driven-core

# Create Pull Request on GitHub
# Title: "Phase 1: Implement protocol-driven core API"
# Description: Link to PHASE_0_SUMMARY.md and COMPATIBILITY_PLAN.md
```

### Phase 9: Code Review & Merge

```bash
# After approval, merge back to main
git checkout main
git pull origin main
git merge --no-ff phase-1/protocol-driven-core

# Push to main
git push origin main

# Optional: Delete the feature branch
git branch -d phase-1/protocol-driven-core
git push origin --delete phase-1/protocol-driven-core
```

---

## Branch Naming Convention

Use this pattern for refactoring branches:

```
phase-N/<description>

Examples:
  phase-1/protocol-driven-core        # Unified API
  phase-2/module-restructuring        # Reorganize modules
  phase-3/optimization                # Performance work
  hotfix/urgent-bug                   # Urgent fixes (from main)
  feature/new-capability              # Feature work
```

---

## Commit Message Template

```
<Type>: <Short summary (50 chars max)>

<Longer description (wrap at 72 chars)>
- Explain what changed and why
- Reference related issues/docs
- Link to relevant planning docs

See docs/developer-guide/<relevant-doc>.md for details.
Fixes #123 (if applicable)
```

### Example

```
Phase 1: Implement protocol-driven core architecture

- Created foodspec.core module with FoodSpec unified API
- Moved baseline functions to core.preprocessing.baseline
- Deprecated old import paths with DeprecationWarning
- Maintained public API surface (all imports still work)
- All tests passing (coverage >= 80%)
- No breaking changes in v1.1.0

See PHASE_0_SUMMARY.md and COMPATIBILITY_PLAN.md for strategy.
Ref: Phase 1 specification
```

---

## Multiple Phases Workflow

For multiple phases (Phase 1, 2, 3, etc.):

```bash
# Phase 1: Create branch, do work, merge to main
git checkout -b phase-1/protocol-driven-core
# ... do work ...
git commit -m "Phase 1: ..."
git push origin phase-1/protocol-driven-core
# Create PR, review, merge to main

# Phase 2: Create branch from updated main
git checkout main
git pull origin main
git checkout -b phase-2/module-restructuring
# ... do work ...
git commit -m "Phase 2: ..."
git push origin phase-2/module-restructuring
# Create PR, review, merge to main

# And so on...
```

---

## Recovery & Rollback

### If Something Goes Wrong

```bash
# View the commit history
git log --oneline -20

# Revert a specific commit (creates new commit)
git revert <commit-hash>

# Reset to previous commit (use with caution!)
git reset --hard <commit-hash>

# View what was deleted (via git reflog)
git reflog
git checkout <lost-commit-hash>
```

### Finding Old Code

```bash
# Search for a function in git history
git log -p -S "function_name" -- src/

# View deleted file
git show <commit-hash>:src/foodspec/old_file.py

# Recover deleted file
git checkout <commit-hash> -- src/foodspec/old_file.py
```

---

## Pre-Merge Checklist

Before merging a refactor branch to main:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage >= 80%: `pytest --cov=src/foodspec --cov-fail-under=80`
- [ ] No unexpected deprecation warnings: `pytest -W error::DeprecationWarning`
- [ ] Type checking passes: `mypy src/ --strict`
- [ ] Code formatted: `ruff format src/`
- [ ] Linting passes: `ruff check src/`
- [ ] Backward compatibility maintained (old imports work)
- [ ] New tests added for refactored code
- [ ] Documentation updated (docs/)
- [ ] RELEASE_NOTES.md updated
- [ ] Commit message clear and references planning docs
- [ ] Code review approved
- [ ] CI/CD pipeline passing

---

## Merge Strategies

### Standard (Recommended)

```bash
# Merge with history preserved
git merge --no-ff phase-1/protocol-driven-core

# Creates a merge commit that shows the branch was merged
# Git log will show all commits from the branch
```

### Squash (for many small commits)

```bash
# Squash all branch commits into one
git merge --squash phase-1/protocol-driven-core
git commit -m "Phase 1: Protocol-driven core (squashed)"

# Clean git history but loses individual commit history
```

### Rebase (keep linear history)

```bash
git rebase main phase-1/protocol-driven-core
git checkout main
git merge fast-forward phase-1/protocol-driven-core

# Linear history but loses the fact that it was a branch
```

**For FoodSpec refactor: Use `--no-ff` (standard merge)** to preserve branch history.

---

## Handling Conflicts

If main has changed while working on your branch:

```bash
# Update main first
git fetch origin
git checkout main
git pull origin main

# Go back to your branch
git checkout phase-1/protocol-driven-core

# Merge main into your branch (resolve conflicts locally)
git merge main
# Resolve conflicts in conflicted files

# Stage resolved files
git add src/foodspec/...

# Complete merge
git commit -m "Merge main into phase-1/protocol-driven-core"

# Push updated branch
git push origin phase-1/protocol-driven-core
```

---

## Summary

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Option A** (Branch + Delete) | History preserved, clean separation, easy review | Requires discipline | ✅ **RECOMMENDED** |
| Option B (Archive in legacy/) | Clear deprecation path | Clutters repo, harder to remove | Gradual migration |
| Main branch refactor | Simplest | Loses history, hard to rollback | ❌ **NOT RECOMMENDED** |

---

## Next Steps

1. **Create Phase 1 branch**: `git checkout -b phase-1/protocol-driven-core`
2. **Build new core** in `src/foodspec/core/`
3. **Add re-exports** in old locations for backward compat
4. **Write tests** including backward compat tests
5. **Commit & push** with clear message referencing Phase 0 docs
6. **Create PR** with detailed description
7. **Get review** ensuring all items in Pre-Merge Checklist
8. **Merge to main** using `--no-ff` flag

---

## References

- [PHASE_0_SUMMARY.md](./PHASE_0_SUMMARY.md) — Phase 0 overview
- [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md) — Backward compatibility strategy
- [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) — Stable APIs to maintain
- [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) — Rules to follow during refactor
- [CONTRIBUTING.md](../../CONTRIBUTING.md) — Contributing guidelines

---

**Created**: 2026-01-24  
**Status**: Active  
**Reviewed**: Phase 0 Complete
