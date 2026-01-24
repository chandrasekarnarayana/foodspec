# Developer Guide Index

**Purpose**: Central hub for FoodSpec development resources  
**Last Updated**: 2026-01-24

---

## 🚀 Getting Started

### First Time Contributing?
1. **Start here**: [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) (5 min read)
2. **Then read**: [CONTRIBUTING.md](../../CONTRIBUTING.md#before-you-start)
3. **Before coding**: Run `ruff check . && mypy src/ && pytest`
4. **Before submitting PR**: Use [PR Checklist](../../CONTRIBUTING.md#pull-request-checklist)

### New to the Codebase?
1. Read [PHASE_0_SUMMARY.md](./PHASE_0_SUMMARY.md) for context
2. Skim [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) for non-negotiables
3. Check [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) to understand stable APIs
4. Review existing code in `src/foodspec/` for patterns

---

## 📚 Core Documentation

### Phase 0: Guardrails & Baseline

| Document | Purpose | Length | Read If... |
|----------|---------|--------|-----------|
| [PHASE_0_SUMMARY.md](./PHASE_0_SUMMARY.md) | Overview of Phase 0 deliverables | 5 min | You're new to the rules |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | 1-page cheat sheet of 7 rules | 5 min | You want a bookmark-able summary |
| [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) | Detailed principles with examples | 30 min | You need detailed guidance |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | How to contribute to FoodSpec | 15 min | You're submitting code |

### Backward Compatibility

| Document | Purpose | Length | Read If... |
|----------|---------|--------|-----------|
| [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md) | Strategy for maintaining backward compat | 20 min | You're refactoring existing APIs |
| [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) | Definitive list of stable APIs | 15 min | You need to know what never breaks |
| [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md) | 8 copy-paste patterns for re-exports | 20 min | You're moving code and need examples |

### Configuration & Tools

| Document | Purpose | Length | Read If... |
|----------|---------|--------|-----------|
| `.ruff.toml` | Ruff linter config | 3 min | You're setting up a dev environment |
| `.pre-commit-config.yaml` | Pre-commit hooks | 5 min | You want local checks before git push |
| `.github/workflows/` | CI/CD configuration | 5 min | You're setting up automated testing |

---

## 🎯 Quick Navigation by Task

### "I want to add a new function"
1. [ENGINEERING_RULES.md#rule-3](./ENGINEERING_RULES.md#rule-3-every-public-functionclass-must-have-docstring--example) — Docstring requirements
2. [ENGINEERING_RULES.md#rule-4](./ENGINEERING_RULES.md#rule-4-every-new-feature-must-include-tests--docs) — Test structure
3. [CONTRIBUTING.md#pull-request-checklist](../../CONTRIBUTING.md#pull-request-checklist) — Before submitting
4. Template: [QUICK_REFERENCE.md#example](./QUICK_REFERENCE.md#example-adding-a-new-function)

### "I need to move/refactor existing code"
1. [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md) — Overall strategy
2. [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) — Is this API stable?
3. [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md) — Copy-paste patterns
4. [ENGINEERING_RULES.md#rule-2](./ENGINEERING_RULES.md#rule-2-no-hidden-global-state) — Avoid introducing global state

### "I'm reviewing a PR"
1. [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) — Quick rule check
2. [CONTRIBUTING.md#pull-request-checklist](../../CONTRIBUTING.md#pull-request-checklist) — Verify checklist
3. [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) — Reference for detailed feedback
4. [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md) — If they're refactoring

### "I'm fixing a bug"
1. [ENGINEERING_RULES.md#rule-7](./ENGINEERING_RULES.md#rule-7-errors-must-be-actionable) — Make errors clear
2. [CONTRIBUTING.md#pull-request-checklist](../../CONTRIBUTING.md#pull-request-checklist) — Verify before merging
3. Don't need to worry about deprecation (bug fixes are compatible!)

### "I'm implementing backward compatibility"
1. [COMPATIBILITY_PLAN.md#deprecation-timeline--versioning](./COMPATIBILITY_PLAN.md#deprecation-timeline--versioning) — Timeline
2. [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md) — 8 ready-to-use patterns
3. [PUBLIC_API_INVENTORY.md#adding-to-public-api](./PUBLIC_API_INVENTORY.md#adding-to-public-api) — Update inventory
4. [BACKWARD_COMPAT_EXAMPLES.md#example-6](./BACKWARD_COMPAT_EXAMPLES.md#example-6-test-for-backward-compatibility) — Test structure

### "I'm writing a migration guide"
1. [BACKWARD_COMPAT_EXAMPLES.md#example-8](./BACKWARD_COMPAT_EXAMPLES.md#example-8-migration-guide-template) — Template
2. [COMPATIBILITY_PLAN.md#user-migration-guide](./COMPATIBILITY_PLAN.md#user-migration-guide) — Strategy
3. Include before/after examples and timeline

---

## 🔍 The 7 Non-Negotiable Rules

**Quick versions** — read [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for full details.

1. **Deterministic Outputs** — Pass `seed` explicitly
2. **No Hidden Global State** — Use dataclasses/pydantic, pass config
3. **Documented Public APIs** — Docstring + type hints + example
4. **Tests + Docs** — ≥80% coverage, tests mirror source structure
5. **Metadata Validated Early** — pydantic models with validators
6. **Pipelines Serializable** — `.to_dict()` / `.from_dict()` methods
7. **Errors Actionable** — What + why + how to fix

---

## 🛠 Common Tasks & Commands

### Setup
```bash
cd FoodSpec
pip install -e ".[dev]"
pre-commit install  # Optional but recommended
```

### Before Committing
```bash
ruff format src/ tests/
ruff check src/ tests/ --fix
mypy src/ --strict
pytest tests/ --cov=src/foodspec
```

### Running Tests
```bash
pytest tests/ -v                                # All tests
pytest tests/test_module.py -v                  # Specific file
pytest tests/test_module.py::TestClass -v       # Specific class
pytest tests/test_module.py::TestClass::test_x  # Specific test
pytest --cov=src/foodspec --cov-report=html    # Coverage report
```

### Check Coverage
```bash
pytest tests/ --cov=src/foodspec --cov-fail-under=80
# Opens htmlcov/index.html in browser if available
```

### Linting & Formatting
```bash
ruff format src/           # Format with Black
ruff check src/ --fix      # Auto-fix issues
mypy src/ --strict         # Type checking (strict mode)
pydocstyle src/            # Check docstrings
```

---

## 📋 Rules Enforcement

### Automated (CI/CD)
- ✅ `ruff format` — Code formatting
- ✅ `ruff check` — Linting (500+ rules)
- ✅ `mypy --strict` — Type checking
- ✅ `pytest --cov` — Tests & coverage (≥80% required)
- ✅ `pydocstyle` — Docstring validation (optional)

### Manual (Code Review)
- ✅ Docstrings present with examples (Rule 3)
- ✅ Seed parameter on probabilistic functions (Rule 1)
- ✅ No hidden global state (Rule 2)
- ✅ Validation at entry point (Rule 5)
- ✅ Error messages actionable (Rule 7)
- ✅ Backward compat maintained (see COMPATIBILITY_PLAN.md)
- ✅ Tests present, ≥80% coverage (Rule 4)

---

## 🔄 Refactor Workflow

When refactoring existing code:

1. **Plan**: Identify which APIs are moving
2. **Check**: [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) — is this API stable?
3. **Migrate**: Use patterns from [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md)
   - If stable: Create re-export wrapper with deprecation warning
   - If experimental: Just move (no compat needed)
4. **Update**: 
   - Update [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md)
   - Update RELEASE_NOTES.md
   - Add migration guide if user-facing
5. **Test**: Add tests from [BACKWARD_COMPAT_EXAMPLES.md#example-6](./BACKWARD_COMPAT_EXAMPLES.md#example-6-test-for-backward-compatibility)
6. **Submit**: Use [PR checklist](../../CONTRIBUTING.md#pull-request-checklist)

---

## 📞 Getting Help

| Question | Resource |
|----------|----------|
| "What are the 7 rules?" | [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) |
| "How do I implement Rule X?" | [ENGINEERING_RULES.md#rule-x](./ENGINEERING_RULES.md) |
| "I'm breaking an existing API, what do I do?" | [COMPATIBILITY_PLAN.md](./COMPATIBILITY_PLAN.md) |
| "Show me a re-export example" | [BACKWARD_COMPAT_EXAMPLES.md](./BACKWARD_COMPAT_EXAMPLES.md) |
| "Which APIs must stay stable?" | [PUBLIC_API_INVENTORY.md](./PUBLIC_API_INVENTORY.md) |
| "What's in my PR checklist?" | [CONTRIBUTING.md#pull-request-checklist](../../CONTRIBUTING.md#pull-request-checklist) |
| "How do I write docstrings?" | [ENGINEERING_RULES.md#rule-3](./ENGINEERING_RULES.md#rule-3-every-public-functionclass-must-have-docstring--example) |
| "How do I write tests?" | [ENGINEERING_RULES.md#rule-4](./ENGINEERING_RULES.md#rule-4-every-new-feature-must-include-tests--docs) |

---

## 📖 Document Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                  CONTRIBUTING.md (Updated)                 │
│         Quick rules + PR checklist for contributors        │
└────────────────────┬────────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    v                v                v
┌──────────────┐  ┌──────────────┐  ┌────────────────┐
│ ENGINEERING  │  │ QUICK        │  │ COMPATIBILITY  │
│ RULES.md     │  │ REFERENCE    │  │ PLAN.md        │
│ (Detailed)   │  │ (1-page)     │  │ (Refactoring)  │
└──────────────┘  └──────────────┘  └────────────────┘
    │                                     │
    │                ┌────────────────────┴─────────────────┐
    │                │                                      │
    v                v                                      v
┌──────────────────────────────────┐  ┌──────────────────────────────┐
│ BACKWARD_COMPAT_EXAMPLES.md      │  │ PUBLIC_API_INVENTORY.md      │
│ (8 copy-paste patterns)          │  │ (Definitive stable APIs)     │
└──────────────────────────────────┘  └──────────────────────────────┘
    │                                     │
    └────────────────┬────────────────────┘
                     │
                     v
         ┌───────────────────────┐
         │ PHASE_0_SUMMARY.md    │
         │ (Overview & next      │
         │  steps)               │
         └───────────────────────┘
```

---

## ✅ Checklist for Developers

Before you start coding:
- [ ] Read [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
- [ ] Set up development environment: `pip install -e ".[dev]"`
- [ ] Optional: Install pre-commit hooks: `pre-commit install`
- [ ] Review [ENGINEERING_RULES.md](./ENGINEERING_RULES.md) for your task type
- [ ] Bookmark [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md)

Before submitting PR:
- [ ] ✅ All tests pass: `pytest tests/ -v`
- [ ] ✅ Coverage ≥80%: `pytest --cov=src/foodspec --cov-fail-under=80`
- [ ] ✅ Code formatted: `ruff format src/ tests/`
- [ ] ✅ Linting passes: `ruff check src/ tests/`
- [ ] ✅ Type checks pass: `mypy src/ --strict`
- [ ] ✅ All items in [PR Checklist](../../CONTRIBUTING.md#pull-request-checklist)

---

## 🎓 Learning Path

**For new contributors:**
1. QUICK_REFERENCE.md (5 min)
2. CONTRIBUTING.md (10 min)
3. ENGINEERING_RULES.md Sections 1-2 (15 min)
4. Start with a small PR (add docstring, improve error message, etc.)

**For refactoring work:**
1. COMPATIBILITY_PLAN.md (15 min)
2. PUBLIC_API_INVENTORY.md (10 min)
3. BACKWARD_COMPAT_EXAMPLES.md (20 min)
4. Start refactoring using patterns

**For code reviewers:**
1. QUICK_REFERENCE.md (5 min)
2. CONTRIBUTING.md PR Checklist (3 min)
3. ENGINEERING_RULES.md (reference as needed)
4. BACKWARD_COMPAT_EXAMPLES.md (reference for refactoring PRs)

---

## 📅 Timeline

| Phase | When | Focus | Lead Doc |
|-------|------|-------|----------|
| 0 | Now (Q1 2026) | Establish guardrails | PHASE_0_SUMMARY.md |
| 1 | Q1-Q2 2026 | Protocol-driven core | TBD |
| 2 | Q2-Q3 2026 | Module restructuring | TBD |
| 3 | Q3-Q4 2026 | Optimization & polish | TBD |
| v2.0.0 | Q4 2026 | Breaking release | RELEASE_NOTES_v2.0.0.md |

---

## 🎯 Success Criteria

Phase 0 is successful when:
- ✅ All contributors understand and follow the 7 rules
- ✅ Code reviews consistently check rule compliance
- ✅ CI/CD enforces rules automatically
- ✅ Zero unexpected deprecation warnings in main branch
- ✅ All PRs pass checklist before merge
- ✅ Backward compat never broken in v1.x
- ✅ Migration paths clear for all breaking changes

---

## 📞 Questions or Feedback?

- **Technical question?** Open a GitHub issue
- **Suggestion for docs?** Open a PR
- **Direct contact?** chandrasekarnarayana@gmail.com

---

## Document Statistics

| Document | Lines | Words | Read Time |
|----------|-------|-------|-----------|
| QUICK_REFERENCE.md | 200 | 1,200 | 5 min |
| ENGINEERING_RULES.md | 800 | 6,500 | 25 min |
| COMPATIBILITY_PLAN.md | 700 | 5,800 | 20 min |
| BACKWARD_COMPAT_EXAMPLES.md | 600 | 4,500 | 15 min |
| PUBLIC_API_INVENTORY.md | 500 | 3,800 | 15 min |
| PHASE_0_SUMMARY.md | 400 | 3,000 | 10 min |
| **Total** | **3,800** | **25,000** | **90 min** |

---

**Created**: 2026-01-24  
**Status**: Active  
**Maintained by**: FoodSpec Core Team

🚀 **Ready to contribute?** Start with [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)!
