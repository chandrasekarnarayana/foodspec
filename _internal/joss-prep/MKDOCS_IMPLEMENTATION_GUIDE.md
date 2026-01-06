# MkDocs Navigation Architecture for JOSS Review

**Goal:** Streamline documentation nav from 96+ entries to ~48 entries, optimizing for JOSS reviewer experience while keeping all content searchable.

---

## 📊 Quick Stats

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Nav entries | 96+ | 48 | -50% |
| Top-level sections | 15 | 7 | -53% |
| Reviewer entry points | 6 highlighted | 6 clear paths | ✓ |
| Files deleted | 0 | 0 | ✓ Safe |
| Files renamed | 0 | 0 | ✓ Safe |

---

## 🎯 Top 6 Reviewer Entry Points

After scrolling past Home/Examples, reviewers see:

1. **Getting Started** — Installation + 15-min quickstart
2. **Workflows** — Oil authentication, QC, mixtures (real examples)
3. **Methods** — Preprocessing, validation, statistics (scientific rigor)
4. **API Reference** — Code structure (engineering review)
5. **Theory** — Spectroscopy, chemometrics (background)
6. **Help & Docs** — Troubleshooting, reproducibility (support)

---

## 📋 Navigation Changes

### ✂️ Sections Removed from Nav (Files Kept)

| Section | Why Removed | How to Find |
|---------|-------------|------------|
| Tutorials (all) | Redundant with Workflows | Search: "tutorials" |
| User Guide (non-workflow) | Detailed CLI/protocol docs for advanced users | Search: "CLI", "YAML", "logging" |
| Developer Guide | For contributors, not JOSS reviewers | Search: "contributing", "plugins" |
| Advanced Topics (05-*) | Internal design, deployment info | Search: "deployment", "schema" |
| Internal (_internal/) | Project history, audit logs | Git history only |
| Reference (supporting) | Statistical tables, method comparisons | Search: "method comparison" |

### ✅ Sections Added to Nav

| Section | Why Added | Location |
|---------|-----------|----------|
| End-to-End Pipeline | Shows complete workflow start→finish | Workflows |
| Reproducibility Guide | JOSS critical: ensures replicability | Help & Docs |
| Reproducibility Checklist | Practical checklist for reviewers | Help & Docs |
| Data Format Reference | Reviewers need to understand data model | Help & Docs |

### 🔄 Consolidations

- **Help sections:** 4 scattered sections → 1 "Help & Docs"
- **Oil auth:** Moved to top of Workflows (most common use case)
- **Statistics:** Subsection under Methods (not standalone)

---

## 📁 File Structure (Unchanged)

All files remain in their current locations:

```
docs/
├── getting-started/          ← In nav
├── workflows/                ← In nav (expanded)
├── methods/                  ← In nav (reorganized)
├── api/                       ← In nav
├── theory/                    ← In nav
├── reference/                 ← Partially in nav
├── tutorials/                 ← NOT in nav (but searchable)
├── user-guide/               ← NOT in nav (but searchable)
├── developer-guide/          ← NOT in nav (but searchable)
├── 05-advanced-topics/       ← NOT in nav (but searchable)
├── _internal/                ← NOT in nav (not searchable)
├── help/                      ← Consolidated
└── troubleshooting/          ← Reorganized
```

**Key:** No files moved or deleted. Only nav visibility changed.

---

## 🚀 Implementation Steps

### Step 1: Backup Current Nav
```bash
cd /home/cs/FoodSpec
git diff mkdocs.yml > mkdocs_nav_changes.patch  # Optional backup
git status  # Ensure clean working directory
```

### Step 2: Apply New Nav
Choose one:

**Option A: Manual Edit**
1. Open `mkdocs.yml`
2. Find the `nav:` section (line ~163)
3. Replace entire `nav:` block with content from `MKDOCS_NAV_PASTE_THIS.md`
4. Save

**Option B: Programmatic (Python)**
```python
import yaml

# Load current mkdocs.yml
with open('mkdocs.yml', 'r') as f:
    config = yaml.safe_load(f)

# Get new nav from MKDOCS_NAV_PASTE_THIS.md (or define inline)
# config['nav'] = [...]  # Replace with new structure

# Save
with open('mkdocs.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```

### Step 3: Validate
```bash
# Check YAML syntax
mkdocs build --strict

# Expected output:
# INFO - Building documentation to directory: /home/cs/FoodSpec/site
# INFO - Documentation built in 20-21 seconds
```

### Step 4: Test Locally
```bash
mkdocs serve
# Visit http://localhost:8000
# Check each nav section loads without 404
```

### Step 5: Commit
```bash
git add mkdocs.yml
git commit -m "refactor(docs): streamline nav for JOSS reviewers

- Reduced nav from 96+ to ~48 entries
- Organized by reviewer priority: Getting Started → Workflows → Methods → API → Theory → Help
- Archived tutorials/user-guide/developer-guide from nav (still searchable)
- Added reproducibility guide and end-to-end pipeline to Help & Docs
- All files kept; only nav visibility changed

This addresses JOSS review priorities:
✓ Quick start path is clear
✓ Real-world examples prominent (Workflows)
✓ Scientific rigor visible (Methods + Theory)
✓ API documented for code review
✓ Reproducibility emphasized"

git log --oneline -1  # Verify commit
```

---

## ⚠️ Rollback Plan

### Immediate Rollback (No Questions Asked)
```bash
git checkout mkdocs.yml
mkdocs build --strict  # Verify
```

### Automated Rollback Script
```bash
bash rollback_mkdocs_nav.sh
```

### If You Want to Tweak Before Rolling Back
```bash
# Make edits to mkdocs.yml
nano mkdocs.yml

# Test incrementally
mkdocs build --strict

# If it breaks, check:
# 1. YAML indentation (must be 2 spaces, no tabs)
# 2. File paths (exact match, case-sensitive on Linux)
# 3. Duplicate entries (grep -n ":" mkdocs.yml | sort | uniq -d)

# Once working:
git add mkdocs.yml
git commit -m "refactor: [your message]"

# If still broken:
git reset HEAD~1  # Undo last commit (keeps file edits)
git checkout mkdocs.yml  # Restore original
```

---

## 🧪 Testing Checklist

Before committing, verify:

- [ ] `mkdocs build --strict` completes without errors
- [ ] No 404 in build output
- [ ] `mkdocs serve` starts on http://localhost:8000
- [ ] Click each top-level nav item (Home, Getting Started, Workflows, Methods, API, Theory, Help)
- [ ] Workflows → Oil Authentication → Complete Example loads
- [ ] Methods → Validation → Cross-Validation & Leakage loads
- [ ] API Reference → Overview loads
- [ ] Search bar works (try "oil authentication", "preprocessing", "leakage")
- [ ] Archived content findable via search (try "tutorials", "CLI", "plugins")

---

## 🔍 Verification Commands

```bash
# 1. Check YAML syntax is valid
python -m yaml mkdocs.yml

# 2. Build with strict mode
mkdocs build --strict

# 3. Count nav entries (before vs after)
grep -c ":" mkdocs.yml  # Quick count

# 4. Verify no broken links
# (mkdocs build --strict already does this)

# 5. Verify files still exist
ls -la docs/tutorials/
ls -la docs/user-guide/
ls -la docs/developer-guide/

# 6. Search index includes archived docs
grep -r "tutorial" docs/  # Should find tutorials/*.md
```

---

## 📚 Design Philosophy

### Reviewer-Centric
Navigation assumes reviewers ask:
1. "How do I get started?" → Getting Started
2. "Does this work in practice?" → Workflows
3. "Is the science sound?" → Methods + Theory
4. "Is the code well-designed?" → API
5. "What if I need help?" → Help & Docs

### Search-First for Advanced Topics
Users who need CLI docs, plugins, or tutorials:
- Use search (fast, exact match)
- Don't clutter main nav

### Preservation Over Deletion
- No files deleted
- No files moved
- All content remains on disk
- All content searchable

### Easy Rollback
- Single file edited (mkdocs.yml)
- Git-backed for safety
- One-command restore

---

## 📖 Related Files

| File | Purpose |
|------|---------|
| `MKDOCS_NAV_PASTE_THIS.md` | Exact YAML to paste into mkdocs.yml |
| `MKDOCS_NAVIGATION_PROPOSAL.md` | Full design rationale and analysis |
| `rollback_mkdocs_nav.sh` | Automated rollback script |
| `.gitignore` (existing) | Ensures site/ is not committed |

---

## ❓ FAQ

**Q: Will JOSS reviewers be able to find everything?**  
A: Yes. All content is:
1. In the nav (if in main path), or
2. Searchable (tutorials, user guide, developer guide)

**Q: What if a file path is wrong?**  
A: `mkdocs build --strict` will fail with exact error. Fix and retry.

**Q: Can we undo this?**  
A: Yes, one command: `git checkout mkdocs.yml`

**Q: Do we need to update links within docs?**  
A: No. Links within `.md` files are file-based, not nav-based. They work either way.

**Q: Will search still find tutorials?**  
A: Yes. Search indexes all `.md` files, regardless of nav.

---

## 🎓 Summary

**For JOSS reviewers:** Cleaner, focused navigation (7 sections instead of 15).  
**For users:** Everything still discoverable via search.  
**For maintainers:** Safe, reversible change to one file.

**Timeline:** ~5 minutes to apply, ~10 minutes to test and commit.

