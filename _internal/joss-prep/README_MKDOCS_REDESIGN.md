# MkDocs Navigation Redesign - Complete Package

## 📦 What You Have

Five documents to implement mkdocs navigation redesign for JOSS:

### 1. ⭐ **START HERE: MKDOCS_YAML_TO_PASTE.md**
   - **Purpose:** Exact YAML block ready to copy/paste
   - **Time:** 2 minutes
   - **Action:** Copy content, paste into mkdocs.yml, save

### 2. 📋 **MKDOCS_IMPLEMENTATION_GUIDE.md**
   - **Purpose:** Step-by-step implementation instructions
   - **Time:** 5-10 minutes
   - **Includes:** Backup, validation, testing, commit commands

### 3. 📊 **MKDOCS_SUMMARY.md**
   - **Purpose:** Executive summary (this document)
   - **Time:** 3 minutes
   - **Audience:** Decision makers, project leads

### 4. 📖 **MKDOCS_NAVIGATION_PROPOSAL.md**
   - **Purpose:** Full design rationale and analysis
   - **Time:** 10-15 minutes
   - **Includes:** Design philosophy, benefits, rollback plan

### 5. 🚀 **MKDOCS_QUICKREF.md**
   - **Purpose:** Quick visual reference
   - **Time:** 2 minutes
   - **Use:** Quick lookup of what changed

### 6. 🛠️ **rollback_mkdocs_nav.sh**
   - **Purpose:** Automated rollback script
   - **Time:** 1 minute
   - **Use:** If you need to revert: `bash rollback_mkdocs_nav.sh`

---

## 🎯 The Goal

Streamline FoodSpec documentation navigation to:
1. Show the **top 6 reviewer-relevant entry points**
2. **Hide internal pages** from nav (but keep them searchable)
3. **Avoid duplicate nav entries**
4. **Keep everything safe** (no files deleted, one-command rollback)

---

## 📊 What Changes

| Metric | Before | After |
|--------|--------|-------|
| **Nav entries** | 96+ | 48 |
| **Top sections** | 15 | 7 |
| **Complexity** | ⚠️ Overwhelming | ✅ Clear |
| **Reviewer path** | 🤔 Unclear | ✅ Obvious |

---

## 🎬 The Six Entry Points

After Home/Examples, reviewers see (in order):

1. **Getting Started** — Installation, quickstart, FAQ
2. **Workflows** ⭐ — Real examples (oil auth, QC, mixtures)
3. **Methods** — Preprocessing, validation, statistics
4. **API Reference** — Code structure, modules
5. **Theory** — Spectroscopy, chemometrics, FAIR
6. **Help & Docs** — Troubleshooting, **reproducibility**, changelog

---

## ✅ What's Safe

- ✅ Only `mkdocs.yml` file modified (nav section only)
- ✅ No files deleted or renamed
- ✅ All doc content remains intact and searchable
- ✅ All internal links still work
- ✅ One-command rollback: `git checkout mkdocs.yml`

---

## 🔄 What Gets Archived (Still Searchable)

Removed from nav but kept on disk and searchable:

- Tutorials (beginner/intermediate/advanced)
- User Guide (CLI, protocols, logging, etc.)
- Developer Guide (for contributors)
- Advanced Topics (internal design)
- Internal (_internal/ folder)

**Access via:** Search bar (try "tutorials", "CLI", "plugins")

---

## 📋 Quick Checklist

- [ ] Read `MKDOCS_YAML_TO_PASTE.md` (2 min)
- [ ] Read `MKDOCS_IMPLEMENTATION_GUIDE.md` (5 min)
- [ ] Decide: Apply changes? (1 min)
- [ ] Apply: Copy/paste YAML (2 min)
- [ ] Test: `mkdocs build --strict` (2 min)
- [ ] Test: `mkdocs serve` and click each section (3 min)
- [ ] Commit: `git add mkdocs.yml` (1 min)

**Total time:** ~15 minutes

---

## 🚀 Implementation (TL;DR)

```bash
# 1. Backup (optional)
git status mkdocs.yml

# 2. Edit mkdocs.yml
# - Find: nav:
# - Replace entire nav section with content from MKDOCS_YAML_TO_PASTE.md
# - Save

# 3. Validate
mkdocs build --strict

# 4. Test
mkdocs serve
# Visit http://localhost:8000 and verify sections load

# 5. Commit
git add mkdocs.yml
git commit -m "refactor(docs): streamline nav for JOSS reviewers"

# 6. (If problems) Rollback
git checkout mkdocs.yml
```

---

## ⚠️ Rollback (If Needed)

**One command:**
```bash
git checkout mkdocs.yml
```

**Or automated:**
```bash
bash rollback_mkdocs_nav.sh
```

---

## 📚 Design Highlights

### Reviewer-Centric
- Clear path: Home → Examples → Getting Started → Workflow → Methods → Theory
- No deep nesting
- No information overload

### Reproducibility-Forward
- Reproducibility guide added to "Help & Docs"
- Reproducibility checklist included
- Shows JOSS that FoodSpec takes this seriously

### Safe & Reversible
- Single file change
- Git-backed
- Search still finds everything
- All links still work

---

## 🎓 Why This Matters for JOSS

JOSS reviewers ask:
1. **"How do I get started?"** → Getting Started section (clear path)
2. **"Does this work in practice?"** → Workflows section (prominent, real examples)
3. **"Is the science sound?"** → Methods + Theory (rigor visible)
4. **"Is the code quality good?"** → API Reference (structure visible)
5. **"Is it reproducible?"** → Help & Docs (reproducibility checklist visible)
6. **"Where do I go for help?"** → Help & Docs (single section, not scattered)

This navigation design **directly supports** JOSS review criteria.

---

## 📞 FAQ

**Q: Will anything break?**  
A: No. Only nav visibility changes. Files stay. Links work. Search finds everything.

**Q: Can we undo this?**  
A: Yes, one command: `git checkout mkdocs.yml`

**Q: Will reviewers find what they need?**  
A: Yes. Core path is clear. Advanced topics searchable.

**Q: Do we need to update internal links?**  
A: No. Links within docs are file-based, not nav-based.

**Q: What if a file path is wrong?**  
A: `mkdocs build --strict` will fail with exact error.

---

## 📖 Next Steps

1. **Choose your path:**
   - **Fast:** Read this file, then `MKDOCS_YAML_TO_PASTE.md`, copy/paste, done
   - **Thorough:** Read all documents, understand design, then apply

2. **Apply changes:**
   - Use `MKDOCS_YAML_TO_PASTE.md`

3. **Test:**
   - Follow `MKDOCS_IMPLEMENTATION_GUIDE.md`

4. **Commit or rollback:**
   - Both are one-command operations

---

## 🎁 Files in This Package

| File | Purpose | Read Time |
|------|---------|-----------|
| **MKDOCS_YAML_TO_PASTE.md** | Copy/paste YAML | 2 min |
| **MKDOCS_IMPLEMENTATION_GUIDE.md** | Step-by-step | 5 min |
| **MKDOCS_SUMMARY.md** | Executive summary | 3 min |
| **MKDOCS_NAVIGATION_PROPOSAL.md** | Full design | 10 min |
| **MKDOCS_QUICKREF.md** | Visual reference | 2 min |
| **rollback_mkdocs_nav.sh** | Automated rollback | N/A |

---

## ✨ Summary

**From:** Overwhelming 96+ nav entries across 15 sections  
**To:** Clear 48 entries across 7 sections  
**For:** JOSS reviewers who need to assess software quality quickly  
**With:** Zero risk, full search access, one-command rollback  

**Status:** Ready to implement ✅

---

## 📞 Support

See the specific documents for more details:
- Confused about YAML? → `MKDOCS_IMPLEMENTATION_GUIDE.md`
- Want to understand the design? → `MKDOCS_NAVIGATION_PROPOSAL.md`
- Need a quick look? → `MKDOCS_QUICKREF.md`
- Ready to apply? → `MKDOCS_YAML_TO_PASTE.md`

