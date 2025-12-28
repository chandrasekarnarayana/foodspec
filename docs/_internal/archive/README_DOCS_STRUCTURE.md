# FoodSpec Documentation Structure

**Last Updated:** December 25, 2025

This document explains the canonical documentation structure for the FoodSpec project.

---

## Documentation Philosophy

**Single Source of Truth:** All documentation lives in `docs/` with carefully selected files at the repository root for GitHub/PyPI visibility.

---

## Root-Level Documentation (GitHub Visibility)

These files remain at the repository root for discoverability:

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Project overview, quick start | All users, first-time visitors |
| `CHANGELOG.md` | Version history, release notes | Users tracking changes |
| `CONTRIBUTING.md` | Contribution guidelines | Contributors |
| `CODE_OF_CONDUCT.md` | Community standards | All community members |
| `LICENSE` | Legal terms (MIT) | Users, legal review |
| `REFACTORING_PLAN.md` | Active refactoring work | Maintainers (temporary) |

**Note:** Root docs should be concise. Detailed guides live in `docs/`.

---

## docs/ Directory Structure

### User-Facing Documentation

```plaintext
docs/
├── index.md                          # Landing page with quick navigation
│
├── 01-getting-started/               # New user onboarding
│   ├── installation.md
│   ├── quickstart_cli.md
│   ├── quickstart_python.md
│   └── quickstart_protocol.md
│
├── 02-tutorials/                     # Hands-on learning paths
│   ├── oil_authentication.md
│   ├── heating_degradation.md
│   └── hyperspectral_analysis.md
│
├── 03-cookbook/                      # Solution recipes
│   ├── preprocessing_recipes.md
│   ├── chemometrics_recipes.md
│   └── workflow_recipes.md
│
├── 04-user-guide/                    # Feature documentation
│   ├── cli.md
│   ├── preprocessing_guide.md
│   ├── chemometrics_guide.md
│   ├── model_registry.md
│   └── libraries.md
│
├── 05-advanced-topics/               # Deep dives
│   ├── advanced_deep_learning.md
│   ├── hsi_and_harmonization.md
│   ├── ftir_raman_preprocessing.md
│   ├── MOATS_IMPLEMENTATION.md       # Matrix correction, calibration transfer
│   └── protocol_benchmarks.md
│
├── 06-developer-guide/               # For contributors
│   ├── RELEASING.md                  # Release process
│   ├── RELEASE_CHECKLIST.md          # Pre-release verification
│   ├── contributing.md
│   ├── architecture.md
│   ├── testing_coverage.md
│   └── design_overview.md
│
└── 07-theory-and-background/         # Scientific foundations
    ├── chemometrics_theory.md
    ├── spectroscopy_basics.md
    └── validation_methods.md
```

### Technical Reference

```plaintext
docs/
├── api/                              # Auto-generated API docs
│   ├── core.md
│   ├── preprocessing.md
│   ├── chemometrics.md
│   └── ...
│
├── protocols/                        # Protocol engine documentation
│   └── protocols_overview.md
│
├── workflows/                        # Workflow system docs
│   └── ...
│
├── visualization/                    # Plotting and viz docs
│   └── ...
│
└── datasets/                         # Example datasets
    └── ...
```

### Maintenance

```plaintext
docs/
├── archive/                          # Historical documents (DO NOT DELETE)
│   ├── FEATURE_AUDIT.md
│   ├── IMPLEMENTATION_AUDIT.md
│   ├── PHASE0_DISCOVERY_REPORT.md
│   ├── PROJECT_STRUCTURE_AUDIT.md
│   ├── CODEBASE_STATUS_SUMMARY.md
│   ├── AUDIT_DOCUMENTATION_INDEX.md
│   ├── CLI_REFACTORING_COMPLETE.md
│   └── [other historical docs]
│
└── assets/                           # Images, CSS, media
    ├── logo.png
    └── ...
```

**Archived Documents:** All files in `docs/archive/` have an "ARCHIVED" banner at the top. These documents are preserved for historical reference but are not current.

---

## Documentation Standards

### Writing Guidelines

1. **Markdown Format:** All docs use standard Markdown (.md)
2. **Code Examples:** Use syntax highlighting (```python)
3. **Cross-References:** Use relative links within docs/
4. **Version Notes:** Include version compatibility where relevant
5. **Examples:** Provide runnable code snippets

### File Naming

- Use lowercase with underscores: `preprocessing_guide.md`
- Be descriptive: `quickstart_cli.md` not `quick.md`
- Prefix numbered sections: `01-getting-started/`

### Navigation (mkdocs.yml)

All user-facing documentation must be included in `mkdocs.yml` navigation. API docs are auto-generated.

---

## Finding Documentation

### For New Users
Start here: `README.md` → `docs/01-getting-started/installation.md`

### For CLI Users
Go to: `docs/04-user-guide/cli.md`

### For Python API Users
Go to: `docs/01-getting-started/quickstart_python.md` → `docs/api/`

### For Contributors
Go to: `CONTRIBUTING.md` → `docs/06-developer-guide/`

### For Scientific Background
Go to: `docs/07-theory-and-background/`

---

## Building Documentation

```bash
# Install mkdocs
pip install mkdocs mkdocs-material

# Serve locally (with live reload)
mkdocs serve

# Build static site
mkdocs build

# Output will be in site/ (gitignored)
```

---

## Maintaining Documentation

### Adding New Documentation

1. Choose the appropriate section (01-07)
2. Create the markdown file with clear naming
3. Add entry to `mkdocs.yml` navigation
4. Cross-reference from related docs
5. Test with `mkdocs serve`

### Archiving Old Documentation

1. Move file to `docs/archive/`
2. Add "ARCHIVED" banner at the top
3. Remove from `mkdocs.yml` navigation
4. Update cross-references

### Updating Documentation

1. Edit the markdown file
2. Update version compatibility notes if needed
3. Verify internal links still work
4. Test with `mkdocs serve`

---

## Documentation Principles

1. **Single Source of Truth:** Don't duplicate content across files
2. **User-Centric:** Write for the reader's goal, not the code structure
3. **Progressive Disclosure:** Basic → intermediate → advanced
4. **Runnable Examples:** All code examples should work as-is
5. **Versioned:** Note breaking changes and version requirements
6. **Searchable:** Use clear headings and keywords

---

## Questions?

- Documentation issues: Open a GitHub issue
- Contributing docs: See `CONTRIBUTING.md`
- Style questions: Follow existing conventions in similar docs

---

**Maintained by:** FoodSpec Core Team  
**Structure Version:** 2.0 (Post-December 2025 Refactor)
