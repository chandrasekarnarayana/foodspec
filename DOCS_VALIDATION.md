# Documentation Quality Tools - Quick Start

This document provides a quick reference for maintainers reviewing documentation PRs.

---

## 🚀 Quick Validation (Before Merging)

**Run this command before merging any documentation PR:**

```bash
python scripts/validate_docs.py
```

**Expected output:**
```
🎉 ALL CHECKS PASSED (4/4)
Documentation is ready for publication!
```

If checks fail, review errors and request changes from contributor.

---

## 📋 What Gets Checked

### 1. Markdown Linting (Optional)
- Heading hierarchy (H1 → H2 → H3, no skipping)
- Consistent list style (use `-` not `*`)
- One H1 per document
- **Requires:** `npm install -g markdownlint-cli`

### 2. Link Validation (Required)
- Broken internal links (missing .md files)
- Broken image links (missing images)
- Invalid anchor links (with `--full` flag)
- Missing alt text (warning only)

### 3. MkDocs Build (Required)
- Runs `mkdocs build --strict`
- Catches syntax errors, broken navigation
- Ensures docs will deploy correctly

### 4. Style Checks (Warnings)
- Code blocks without language tags
- Headings with trailing periods

---

## 🛠️ Installation

### Required (Python)
```bash
pip install mkdocs mkdocs-material mkdocstrings
```

### Optional (Node.js)
```bash
npm install -g markdownlint-cli
```

---

## 📖 Commands

| Task | Command |
|------|---------|
| **Full validation** | `python scripts/validate_docs.py` |
| **Fast validation** | `python scripts/validate_docs.py --skip-build` |
| **With anchor checking** | `python scripts/validate_docs.py --full` |
| **Links only** | `python scripts/check_docs_links.py` |
| **Preview locally** | `mkdocs serve` (http://localhost:8000) |
| **Build for deployment** | `mkdocs build` |

---

## 📚 Detailed Guides

For comprehensive instructions, see:

- **[Documentation Style Guide](docs/06-developer-guide/documentation_style_guide.md)** — Writing standards, templates, "Definition of Done"
- **[Documentation Maintainer Guide](docs/06-developer-guide/documentation_maintainer_guide.md)** — PR review workflow, troubleshooting
- **[Scripts README](scripts/README_DOCS_VALIDATION.md)** — Validation tool details, CI/CD setup

---

## 🐛 Common Issues

### Issue: "ModuleNotFoundError: No module named 'mkdocs'"
**Fix:** `pip install mkdocs mkdocs-material mkdocstrings`

### Issue: "markdownlint: command not found"
**Fix (optional):** `npm install -g markdownlint-cli`  
**Or:** Skip markdownlint (validation still works)

### Issue: "Permission denied" when running scripts
**Fix:** `chmod +x scripts/*.py`

### Issue: Validation passes locally but fails in CI
**Causes:**
- Uncommitted files (check `git status`)
- Case-sensitive paths (use lowercase filenames)
- Line endings (Windows: `git config core.autocrlf input`)

---

## ✅ Pre-Merge Checklist

- [ ] Run `python scripts/validate_docs.py` → All checks pass
- [ ] Preview locally (`mkdocs serve`) → New pages render correctly
- [ ] Check mobile rendering (resize browser to 375px width)
- [ ] Test code examples (copy-paste and run)
- [ ] Verify images load and have alt text
- [ ] Confirm page added to `mkdocs.yml` navigation

---

## 🚢 Deployment

**Automatic on push to `main`:**
- GitHub Actions runs `mkdocs build`
- Deploys to: https://chandrasekarnarayana.github.io/foodspec/
- Wait ~2-3 minutes for deployment

**Rollback if needed:**
```bash
git revert <commit-hash>
git push origin main
```

---

## 📞 Support

**Questions?**
- **GitHub Issues:** [foodspec/issues](https://github.com/chandrasekarnarayana/foodspec/issues) (label: `documentation`)
- **Maintainer:** @chandrasekarnarayana

---

**Last updated:** December 28, 2024
