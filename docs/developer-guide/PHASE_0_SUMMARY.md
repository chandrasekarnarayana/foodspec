# Phase 0: Guardrails & Baseline — Summary

**Status**: ✅ Complete  
**Created**: 2026-01-24  
**Scope**: Engineering rules + backward compatibility strategy for FoodSpec refactor

---

## What Was Created

This phase establishes the **engineering foundation** for refactoring FoodSpec into a protocol-driven framework while maintaining full backward compatibility.

### 📋 Documents Created

| Document | Purpose | Audience | Key Takeaway |
|----------|---------|----------|--------------|
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | **Updated** with detailed engineering rules | Contributors | 7 non-negotiables + PR checklist |
| [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) | Codifies non-negotiable principles | Core team, reviewers | Detailed rules with examples & anti-patterns |
| [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md) | Backward compatibility strategy | Maintainers | Deprecation timeline + re-export patterns |
| [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md) | Ready-to-use implementation templates | Developers | 8 copy-paste patterns for migrations |
| [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) | Definitive list of stable APIs | Core team | What must never break in v1.x |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | One-page cheat sheet | Daily use | 7 rules in digestible format |

---

## The 7 Non-Negotiable Rules

Every contribution **must** follow these:

### 1. ✅ **Deterministic Outputs**
- Pass `seed` explicitly to all probabilistic functions
- Use `np.random.default_rng(seed)` in NumPy
- **Test**: Identical seeds → identical outputs

### 2. ✅ **No Hidden Global State**
- No module-level mutable defaults
- Use dataclasses/pydantic for config
- Pass config as explicit parameter

### 3. ✅ **Every Public API Has Docstring + Example**
- NumPy-style docstrings required
- Type hints mandatory
- Runnable examples in docstring

### 4. ✅ **Tests + Docs Required**
- ≥80% code coverage for new code
- Tests mirror source structure: `src/foo.py` ↔ `tests/test_foo.py`
- Documentation in `docs/` updated

### 5. ✅ **Metadata Validated Early**
- Use pydantic models with validators
- Raise `ValueError` immediately at entry point
- Never defer validation

### 6. ✅ **Pipelines Serializable**
- Config as dataclass/pydantic (not plain dict)
- `.to_dict()` / `.from_dict()` methods
- JSON/YAML compatible

### 7. ✅ **Errors Must Be Actionable**
- Error message: what + why + how to fix
- Specific exception types (ValueError, TypeError, etc.)
- Clear suggestions and context

---

## Backward Compatibility Strategy

### Core Principle
**Old user code continues to work**, but with optional deprecation warnings guiding migration.

### Timeline

```
v1.0.0 (Current)
  ├─ Original API fully functional
  └─ No warnings

v1.1.0 (Q1 2026)
  ├─ NEW: Refactored core available
  ├─ OLD: Original API still works
  └─ DEPRECATED: Moved functions emit warnings

v1.2.0 - v1.9.0 (Q2-Q3 2026)
  ├─ MORE: Internal restructuring
  ├─ OLD: Original API still works
  └─ SAME: Deprecation warnings continue

v2.0.0 (Q4 2026, BREAKING)
  ├─ NEW: Clean, modern structure
  ├─ OLD: Deprecated APIs removed
  └─ MIGRATION: Guide provided in RELEASE_NOTES_v2.0.0.md
```

### Re-export Patterns

**Pattern 1: Simple re-export (old location keeps working, no warning)**
```python
# src/foodspec/old_location.py
from foodspec.new_location import function
__all__ = ['function']
```

**Pattern 2: Re-export with deprecation warning**
```python
# src/foodspec/old_location.py
import warnings

def function(*args, **kwargs):
    warnings.warn("...deprecated, use foodspec.new_location.function", 
                  DeprecationWarning, stacklevel=2)
    from foodspec.new_location import function as _impl
    return _impl(*args, **kwargs)
```

### What Must Stay Stable (Public API)

All these remain importable from `foodspec` or sub-modules through v1.x:
- Core classes: `Spectrum`, `FoodSpectrumSet`, `HyperSpectralCube`, `OutputBundle`, `RunRecord`, `FoodSpec`
- I/O: `load_folder`, `load_library`, `create_library`, `load_csv_spectra`, etc.
- Preprocessing: `baseline_als`, `baseline_polynomial`, `baseline_rubberband`, etc.
- QC/Stats: All functions in `foodspec.stats`, `foodspec.qc`, `foodspec.metrics`
- Advanced: All Moat functions (matrix correction, calibration transfer, heating trajectory)
- Data governance: All dataset intelligence functions
- Utilities: Artifact management, plugins, synthetic data generation, etc.

See [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) for the complete definitive list.

---

## How to Use These Documents

### For Developers: Start Here
1. Read [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) (5 min) — understand the 7 rules
2. Bookmark it for daily reference
3. Before coding: `ruff format . && ruff check . && mypy src/`
4. Before PR: Check items in CONTRIBUTING.md checklist

### For Code Reviewers
1. Use [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) to spot violations
2. Reference [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) for detailed guidance
3. Check [PR Checklist](../../CONTRIBUTING.md#pull-request-checklist) before approving

### For Refactoring Tasks
1. Identify APIs being moved
2. Check [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) — is this API stable?
3. If stable, use re-export pattern from [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md)
4. Update PUBLIC_API_INVENTORY.md to reflect new location
5. Add test from [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md#example-6-test-for-backward-compatibility)

### For Release Planning
1. Decide which APIs to move/refactor in this release
2. Document as "deprecated" with v2.0.0 removal date
3. Update RELEASE_NOTES.md with deprecation section
4. Provide migration guide in `docs/migration/`
5. Include in release announcement

---

## Enforcement

### Automated (Tools)
```bash
ruff format .          # Code formatting
ruff check .           # 500+ linting rules
mypy src/ --strict     # Type checking
pytest --cov=src/     # Tests & coverage
```

### Manual (Code Review)
- ✅ Docstrings present with examples
- ✅ Seed parameter on probabilistic functions
- ✅ No hidden global state
- ✅ Validation at entry point
- ✅ Error messages actionable
- ✅ Backward compat maintained

### CI/CD
- All checks must pass before merge
- Coverage must be ≥80% for new code
- No deprecation warnings (except in backward compat tests)

---

## Implementation Checklist

- [x] Updated CONTRIBUTING.md with 7 rules
- [x] Created ENGINEERING_RULES.md (detailed, with examples)
- [x] Created COMPATIBILITY_PLAN.md (deprecation strategy)
- [x] Created BACKWARD_COMPAT_EXAMPLES.md (8 copy-paste patterns)
- [x] Created PUBLIC_API_INVENTORY.md (definitive API surface)
- [x] Created QUICK_REFERENCE.md (1-page cheat sheet)
- [ ] TODO: Create .ruff.toml config file (optional, for enforcement)
- [ ] TODO: Create .pre-commit-config.yaml (optional, for pre-commit hooks)
- [ ] TODO: Create GitHub Actions workflow for CI/CD (optional, if not already present)

---

## Next Steps

### Immediate (Phase 0 completion)
1. ✅ Share documents with core team
2. ✅ Get feedback on rules and timeline
3. ✅ Finalize deprecation dates (v1.1, v2.0)
4. ✅ Optional: Set up pre-commit hooks & ruff config

### Phase 1 (Protocol-Driven Core)
1. Implement new FoodSpec unified API
2. Keep all existing functions working (via re-exports if moved)
3. Add deprecation warnings to any moved functions
4. Update documentation with migration paths

### Phase 2+ (Ongoing Refactoring)
1. Systematically refactor modules to new structure
2. Use re-export patterns from BACKWARD_COMPAT_EXAMPLES.md
3. Write backward compat tests
4. Update PUBLIC_API_INVENTORY.md as structure changes

### v2.0.0 Release
1. Remove all deprecated APIs
2. Clean up re-export modules
3. Publish migration guide
4. Announce in release notes

---

## File Structure

```
docs/developer-guide/
├── ENGINEERING_RULES.md              # Non-negotiables (7 rules, detailed)
├── COMPATIBILITY_PLAN.md             # Backward compat strategy
├── BACKWARD_COMPAT_EXAMPLES.md       # 8 ready-to-use patterns
├── PUBLIC_API_INVENTORY.md           # Definitive stable API list
├── QUICK_REFERENCE.md                # 1-page cheat sheet
└── (Phase 0 summary — this file)

CONTRIBUTING.md                       # Updated with rules + PR checklist
```

---

## FAQ

**Q: What if I find a bug in a public API?**  
A: Fix it! Bug fixes don't require deprecation. Document in RELEASE_NOTES.md as a patch.

**Q: Can I add an optional parameter to a public function?**  
A: Yes! Adding is backward compatible. Removing requires deprecation cycle.

**Q: What counts as a "breaking change"?**  
A: Removing/renaming public APIs, changing function signatures (removing params, changing return type), altering behavior. All require deprecation cycle + major version bump.

**Q: How do I know if my code follows Rule 7 (actionable errors)?**  
A: Read [ENGINEERING_RULES.md#rule-7](./ENGINEERING_RULES.md#rule-7-errors-must-be-actionable). Ask: "Could a user fix this problem from the error message alone?" If no, revise.

**Q: What if I disagree with a rule?**  
A: Open an issue! Rules can evolve, but changes require team discussion.

**Q: Can I use a singleton?**  
A: Rarely. Only with documented justification in code AND in ENGINEERING_RULES.md. Discuss in issue first.

---

## Getting Help

- 📖 **Rules question?** → [ENGINEERING_RULES.md](./ENGINEERING_RULES.md)
- 💡 **How to implement?** → [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md)
- 🚀 **Quick start?** → [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
- 🔄 **Compat question?** → [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md)
- 📋 **Is this API stable?** → [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md)
- 🤝 **Contributing?** → [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## Document Summary Table

| Document | Lines | Key Sections | Best For |
|----------|-------|--------------|----------|
| CONTRIBUTING.md | ~200 | Rules + checklist | Contributors (quick) |
| ENGINEERING_RULES.md | ~800 | 7 rules detailed with code examples | Detailed reference |
| COMPATIBILITY_PLAN.md | ~700 | Timeline + re-export patterns | Refactoring tasks |
| BACKWARD_COMPAT_EXAMPLES.md | ~600 | 8 copy-paste patterns | Implementation |
| PUBLIC_API_INVENTORY.md | ~500 | Stable API list | What never breaks |
| QUICK_REFERENCE.md | ~200 | 1-page cheat sheet | Daily use |

**Total**: ~3,000 lines of guardrails documentation

---

## Version Info

- **Documents Version**: 1.0
- **Created**: 2026-01-24
- **FoodSpec Version**: 1.0.0
- **Compatibility**: v1.0.0+
- **Status**: Active

---

**Questions or suggestions?**  
Open an issue or contact: chandrasekarnarayana@gmail.com

---

## Quick Links

- 📖 [Full ENGINEERING_RULES.md](./ENGINEERING_RULES.md)
- 🔄 [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md)
- 💡 [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md)
- 📋 [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md)
- 🚀 [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
- 🤝 [CONTRIBUTING.md](../../CONTRIBUTING.md)

**Created as part of Phase 0 — Guardrails & Repo Baseline for FoodSpec Refactor** ✅
