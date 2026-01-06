# MkDocs Navigation Redesign - Action Checklist

## ✅ Pre-Implementation Checklist

- [ ] Read `MASTER_INDEX_MKDOCS.md` (understand what you have)
- [ ] Read `MKDOCS_SUMMARY.md` (understand what changes)
- [ ] Read `MKDOCS_YAML_TO_PASTE.md` (understand what to copy)
- [ ] Git status is clean (`git status` shows no uncommitted changes)
- [ ] You're in the FoodSpec root directory
- [ ] You have mkdocs installed (`which mkdocs` or `mkdocs --version`)

---

## 🚀 Implementation Checklist

### Step 1: Backup & Prepare
- [ ] Check current status: `git status mkdocs.yml`
- [ ] Optional: Save a copy: `cp mkdocs.yml mkdocs.yml.backup`
- [ ] Optional: Create a patch: `git diff mkdocs.yml > nav.patch`
- [ ] Open `mkdocs.yml` in your text editor

### Step 2: Find & Replace Nav Section
- [ ] Locate the line starting with `nav:`
- [ ] Locate the line starting with `markdown_extensions:` (this is where nav ends)
- [ ] Select all lines from `nav:` (inclusive) to the line BEFORE `markdown_extensions:` (exclusive)
- [ ] Copy the entire selected text
- [ ] Delete the selected text

### Step 3: Paste New Nav
- [ ] Open `MKDOCS_YAML_TO_PASTE.md`
- [ ] Locate the code block starting with ` ```yaml`
- [ ] Select the entire YAML content (from `nav:` to the last `Versioning: reference/versioning.md`)
- [ ] Copy the content
- [ ] Paste into your mkdocs.yml file at the same location where you deleted
- [ ] Verify indentation looks right (should be 2 spaces per level)

### Step 4: Save & Validate
- [ ] Save the file in your editor
- [ ] Run: `mkdocs build --strict`
- [ ] Check output: Should say "Documentation built in ~20 seconds"
- [ ] If errors: Fix them (check file paths or indentation)

### Step 5: Local Testing
- [ ] Run: `mkdocs serve`
- [ ] Open browser: http://localhost:8000
- [ ] Click each top-level section:
  - [ ] Home
  - [ ] Examples
  - [ ] Getting Started
  - [ ] Workflows
  - [ ] Methods
  - [ ] API Reference
  - [ ] Theory
  - [ ] Help & Docs
- [ ] Test search functionality:
  - [ ] Search: "oil authentication"
  - [ ] Search: "preprocessing"
  - [ ] Search: "leakage"
  - [ ] Search: "tutorials" (should find archived content)

### Step 6: Verify File Paths
- [ ] Run: `ls docs/getting-started/installation.md`
- [ ] Run: `ls docs/workflows/authentication/oil_authentication.md`
- [ ] Run: `ls docs/methods/validation/cross_validation_and_leakage.md`
- [ ] Run: `ls docs/api/core.md`
- [ ] Run: `ls docs/theory/spectroscopy_basics.md`
- [ ] Run: `ls docs/troubleshooting/troubleshooting_faq.md`
- [ ] Run: `ls docs/reproducibility.md`
- [ ] Run: `ls docs/protocols/reproducibility_checklist.md`

### Step 7: Commit
- [ ] Stage changes: `git add mkdocs.yml`
- [ ] Review: `git diff --cached mkdocs.yml`
- [ ] Commit: `git commit -m "refactor(docs): streamline nav for JOSS reviewers"`
- [ ] Verify: `git log --oneline -1` (should show your commit)

---

## 🔄 If Something Breaks

### Option A: Immediate Rollback
- [ ] Run: `git checkout mkdocs.yml`
- [ ] Run: `mkdocs build --strict` (verify it works)
- [ ] Done! You're back to original

### Option B: Automated Rollback
- [ ] Run: `bash rollback_mkdocs_nav.sh`
- [ ] Respond: `y` to the prompt
- [ ] Done! You're back to original

### Option C: Manual Fix
- [ ] Check error message from `mkdocs build --strict`
- [ ] Common issues:
  - File path typo (must match exactly)
  - Wrong indentation (must be 2 spaces, not 4 or tabs)
  - Duplicate entry (check for duplicate nav items)
- [ ] Fix the issue in mkdocs.yml
- [ ] Rerun: `mkdocs build --strict`
- [ ] If still broken, rollback (Option A or B above)

---

## ✨ Post-Implementation Checklist

- [ ] All sections load without 404
- [ ] Search finds content
- [ ] Archived content is searchable (tutorials, CLI, plugins)
- [ ] No errors in build output
- [ ] All links within docs still work
- [ ] Changes committed to git
- [ ] Able to rollback if needed (verified one of the rollback options)

---

## 🎯 Verification Checklist (Detailed)

### Navigation Structure
- [ ] Home page exists and loads
- [ ] Examples gallery loads
- [ ] Getting Started section has 6 items
  - [ ] Overview loads
  - [ ] Installation loads
  - [ ] 15-Minute Quickstart loads
  - [ ] First Steps (CLI) loads
  - [ ] Understanding Results loads
  - [ ] FAQ loads
- [ ] Workflows section has 9 subsections
  - [ ] Overview loads
  - [ ] Oil Authentication → Complete Example loads
  - [ ] Oil Authentication → Domain Templates loads
  - [ ] Quality & Monitoring → Heating Quality loads
  - [ ] Quality & Monitoring → Aging Analysis loads
  - [ ] Quality & Monitoring → Batch QC loads
  - [ ] Quantification → Mixture Analysis loads
  - [ ] Quantification → Calibration loads
  - [ ] Harmonization → Multi-Instrument loads
  - [ ] Harmonization → Calibration Transfer loads
  - [ ] Spatial Analysis → Hyperspectral Mapping loads
  - [ ] End-to-End Pipeline loads
  - [ ] Design & Reporting loads
- [ ] Methods section loads (all subsections)
- [ ] API Reference section loads (all modules)
- [ ] Theory section loads (all topics)
- [ ] Help & Docs section loads (all items)

### Search Functionality
- [ ] Search finds "oil authentication" (in Workflows)
- [ ] Search finds "preprocessing" (in Methods)
- [ ] Search finds "leakage" (in Methods → Validation)
- [ ] Search finds "tutorials" (archived, still searchable)
- [ ] Search finds "CLI" (archived, still searchable)
- [ ] Search finds "plugins" (archived, still searchable)

### Build & Performance
- [ ] `mkdocs build --strict` completes
- [ ] Build time: ~20-21 seconds
- [ ] No 404 errors in output
- [ ] No warnings about broken links
- [ ] `mkdocs serve` starts cleanly
- [ ] Pages load quickly in browser

---

## 📋 Documentation Checklist

Have you read?
- [ ] MASTER_INDEX_MKDOCS.md (overview)
- [ ] MKDOCS_SUMMARY.md (what changes)
- [ ] MKDOCS_IMPLEMENTATION_GUIDE.md (how to do it)
- [ ] MKDOCS_YAML_TO_PASTE.md (what to paste)

Have you saved?
- [ ] Original mkdocs.yml (if you made a backup)
- [ ] The new nav block somewhere safe

---

## 🎬 Ready? Start Here

1. **First time?** → Start with `MASTER_INDEX_MKDOCS.md`
2. **Need quick summary?** → Read `MKDOCS_SUMMARY.md`
3. **Ready to implement?** → Go to "Implementation Checklist" above
4. **Something broke?** → Go to "If Something Breaks" section above

---

## 🆘 Troubleshooting

### Build fails with "File not found"
- Check the file path in mkdocs.yml
- Compare with actual file in docs/ folder
- Verify spelling and case (Linux is case-sensitive)

### Build fails with YAML error
- Check indentation (must be exactly 2 spaces per level)
- Verify no tabs used (only spaces)
- Check for duplicate nav items

### Page shows 404
- Run `mkdocs build --strict` to see all errors
- Check file path in nav matches actual file
- Verify file exists: `ls docs/path/to/file.md`

### Search doesn't find archived content
- This is expected behavior (search indexes based on nav + file system)
- Try search phrase exactly (case-insensitive)
- Archived files are still searchable if they exist on disk

### Can't rollback
- Verify you're in the FoodSpec root directory
- Verify git is installed: `which git`
- Verify mkdocs.yml is tracked by git: `git ls-files mkdocs.yml`
- Try manual rollback: Copy from `.git/HEAD` or contact maintainer

---

## ✅ Final Checklist (Before Declaring Success)

- [ ] Build succeeds: `mkdocs build --strict`
- [ ] No 404 errors
- [ ] All nav sections load
- [ ] Search works
- [ ] Changes committed to git
- [ ] Can rollback if needed
- [ ] Documentation updated (if needed)
- [ ] Team notified (if needed)

---

## 🎉 Done!

You've successfully completed the mkdocs navigation redesign for JOSS!

**Impact:**
- ✅ Nav reduced from 96+ to 48 entries
- ✅ Top sections reduced from 15 to 7
- ✅ Reviewer path is clear
- ✅ All content still accessible
- ✅ Zero risk to users

**Next steps:**
- Share the changes with the team
- Monitor feedback from users
- If issues, rollback is one command away

Thank you for making FoodSpec documentation better! 🎓

