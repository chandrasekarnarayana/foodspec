# MkDocs Navigation Redesign for JOSS - Executive Summary

## 🎯 Objective

Streamline FoodSpec documentation navigation to highlight the top 6 reviewer-relevant entry points while archiving internal pages from nav (but keeping them searchable and intact on disk).

---

## 📊 Impact Summary

| Metric | Current | Proposed | Impact |
|--------|---------|----------|--------|
| **Nav entries** | 96+ | 48 | ✅ -50% reduction |
| **Top-level sections** | 15 | 7 | ✅ Cleaner UX |
| **Max nav depth** | 4-5 levels | 3 levels | ✅ Better UX |
| **Files modified** | - | 1 (mkdocs.yml) | ✅ Safe |
| **Files deleted** | - | 0 | ✅ Nothing lost |
| **Files moved/renamed** | - | 0 | ✅ All links work |

---

## 🚀 The Six Entry Points (In Order)

After scrolling past Home/Examples, reviewers immediately see:

### 1️⃣ **Getting Started**
- Installation
- 15-minute quickstart  
- First steps (CLI)
- Understanding results
- FAQ

**Why:** Essential for all first-time users

---

### 2️⃣ **Workflows** ⭐ Most Important
- Oil authentication (complete example)
- Quality & monitoring (heating, aging, batch QC)
- Quantification (mixtures, calibration)
- Harmonization (multi-instrument)
- Spatial analysis (hyperspectral mapping)
- End-to-end pipeline

**Why:** Real-world examples show software works. Reviewers verify credibility here.

---

### 3️⃣ **Methods**
- Preprocessing (baseline, normalization, etc.)
- Chemometrics (models, classification, PCA)
- Validation (cross-validation, leakage, robustness)
- Statistics (t-tests, ANOVA, hypothesis testing)

**Why:** Reviewers assess scientific rigor. All methods justifications here.

---

### 4️⃣ **API Reference**
- Core modules
- Datasets, preprocessing, chemometrics
- Machine learning, I/O, workflows
- Metrics, statistics

**Why:** Code reviewers verify software architecture and documentation.

---

### 5️⃣ **Theory**
- Spectroscopy basics
- Food applications
- Chemometrics & ML
- RQ engine, harmonization, MOATS
- Data structures & FAIR

**Why:** Scientific foundation. Reviewers check theoretical soundness.

---

### 6️⃣ **Help & Docs**
- Troubleshooting
- Reproducibility guide ⭐ (NEW)
- Reproducibility checklist ⭐ (NEW)
- Data format reference ⭐ (NEW)
- Glossary, changelog, versioning

**Why:** Support + reproducibility emphasis (critical for JOSS).

---

## 🎁 What Gets Archived from Nav (But Stays Searchable)

| Section | Files Remain | Searchable | Why Archived |
|---------|--------------|-----------|--------------|
| Tutorials | ✅ Yes | ✅ Yes | Redundant with Workflows |
| User Guide | ✅ Yes | ✅ Yes | Advanced; users search for what they need (CLI, protocols) |
| Developer Guide | ✅ Yes | ✅ Yes | For contributors, not JOSS reviewers |
| Advanced Topics | ✅ Yes | ✅ Yes | Internal design/deployment docs |
| Internal (_internal/) | ✅ Yes | ❌ No | Project history; in git only |

---

## 🔧 Implementation

### Three-Step Process

**Step 1: Copy New Nav**
- Open `mkdocs.yml`
- Locate `nav:` section
- Replace with content from `MKDOCS_NAV_PASTE_THIS.md`

**Step 2: Validate**
```bash
mkdocs build --strict
# Should complete in ~20 seconds without errors
```

**Step 3: Test**
```bash
mkdocs serve
# Visit http://localhost:8000 and click each section
```

**Time required:** ~5-10 minutes

---

## 🛡️ Safety & Rollback

### No Risk
- ✅ Only `mkdocs.yml` modified (nav section only)
- ✅ All doc files remain intact
- ✅ All links within docs still work
- ✅ All content searchable
- ✅ Single-command rollback: `git checkout mkdocs.yml`

### Verification
```bash
# Before
git status mkdocs.yml

# After applying changes
mkdocs build --strict
git add mkdocs.yml
git commit -m "refactor(docs): streamline nav for JOSS"

# To undo
git checkout mkdocs.yml
```

---

## 📋 Quality Assurance Checklist

- [ ] `mkdocs build --strict` succeeds
- [ ] No 404 errors in build output
- [ ] Each nav section loads in browser
- [ ] Search finds "oil authentication"
- [ ] Search finds archived content ("tutorials", "CLI")
- [ ] All workflow examples work
- [ ] Methods → Validation loads correctly

---

## 💡 Design Philosophy

### Principle 1: Reviewer-Centric Navigation
- Home → Examples → Getting Started → Pick a Workflow → Deep Dive

### Principle 2: Archive Non-Critical Content (Not Delete)
- Tutorials, user guide, dev guide archived from nav
- All remain searchable
- Users searching for "CLI" still find it

### Principle 3: Emphasize Reproducibility
- Added to "Help & Docs"
- Shows JOSS that FoodSpec takes reproducibility seriously

### Principle 4: Safe & Reversible
- Single file change
- Git-backed
- One command to rollback

---

## 📚 Deliverables

Four documents have been created:

| File | Purpose |
|------|---------|
| **MKDOCS_NAV_PASTE_THIS.md** | ⭐ Copy/paste the YAML here |
| **MKDOCS_IMPLEMENTATION_GUIDE.md** | Step-by-step implementation |
| **MKDOCS_NAVIGATION_PROPOSAL.md** | Full design analysis |
| **MKDOCS_QUICKREF.md** | Visual reference (this document) |
| **rollback_mkdocs_nav.sh** | Automated rollback script |

---

## ✅ Benefits for JOSS

✅ **Cleaner first impression**  
- Reviewers see 7 sections, not 15+

✅ **Focused on examples**  
- Workflows section prominent and organized

✅ **Scientific rigor visible**  
- Methods section shows statistical validation
- Theory section shows foundation

✅ **Reproducibility emphasized**  
- Dedicated section in Help & Docs
- Reproducibility checklist prominent

✅ **Code quality visible**  
- API Reference section for code review

✅ **Everything still accessible**  
- No content deleted
- All searchable

✅ **Safe & reversible**  
- Single file, one command to undo

---

## 🎬 Next Steps

1. **Review** the four proposal documents (5 min read)
2. **Decide** if you want to apply the changes
3. **Apply** using `MKDOCS_NAV_PASTE_THIS.md` (3-5 min)
4. **Test** with `mkdocs build --strict` and `mkdocs serve` (2 min)
5. **Commit** if happy, or **rollback** if not (1 command)

---

## 📞 Questions?

- **"Will reviewers find everything?"**  
  Yes. Core path is clear. Advanced topics searchable.

- **"What if something breaks?"**  
  Run: `git checkout mkdocs.yml` (single command rollback)

- **"Are any files deleted?"**  
  No. All files preserved. Just nav visibility changed.

- **"Will old links break?"**  
  No. Links within docs are file-based, not nav-based.

---

## 🎓 Summary

**From:** 96+ nav entries scattered across 15 sections  
**To:** 48 nav entries organized in 7 clear sections  
**For:** JOSS reviewers who need quick access to core functionality  
**With:** Zero risk, full search access, one-command rollback  

**Ready to apply:** Yes ✅

