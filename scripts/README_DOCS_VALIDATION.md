# Documentation Validation Scripts

This directory contains tools for validating FoodSpec documentation quality.

---

## Scripts

### 1. validate_docs.py (Comprehensive)

**Purpose:** Run all documentation validation checks in one command.

**Usage:**
```bash
# Standard validation (fast, ~10 seconds)
python scripts/validate_docs.py

# Full validation with anchor checking (slower, ~30 seconds)
python scripts/validate_docs.py --full

# Skip MkDocs build (for quick link checks)
python scripts/validate_docs.py --skip-build
```

**Checks performed:**
- ✅ Markdown linting (markdownlint, if installed)
- ✅ Link validation (broken links, images, anchors)
- ✅ MkDocs build validation
- ✅ Style checks (code block language tags, heading periods)

**Exit codes:**
- `0` = All checks passed
- `1` = Some checks failed

**Example output:**
```
======================================================================
🚀 FoodSpec Documentation Validation Suite
======================================================================
Documentation root: /home/cs/FoodSpec/docs
Full validation: False

[... check results ...]

======================================================================
📊 VALIDATION SUMMARY
======================================================================
✅ Markdownlint
✅ Link Validation
✅ MkDocs Build
✅ Style Checks
======================================================================
🎉 ALL CHECKS PASSED (4/4)
Documentation is ready for publication!
======================================================================
```

---

### 2. check_docs_links.py (Link Checker)

**Purpose:** Check for broken links, images, and anchors in documentation.

**Usage:**
```bash
# Basic link check (fast)
python scripts/check_docs_links.py

# Include anchor validation (slower)
python scripts/check_docs_links.py --check-anchors
```

**Checks performed:**
- ✅ Broken internal links (missing .md files)
- ✅ Broken image links (missing .png/.svg files)
- ✅ Invalid anchor links (#heading-slug) — with `--check-anchors`
- ⚠️  Missing alt text on images (warning only)

**Example output:**
```
Checking 150 markdown files...

❌ MISSING INTERNAL LINKS (relative to docs/):
   02-tutorials/oil_auth.md -> ../missing_page.md

⚠️  MISSING ALT TEXT (accessibility issue):
   02-tutorials/example.md -> ../assets/plot.png

======================================================================
❌ ERRORS: 1
⚠️  WARNINGS: 1
======================================================================
```

---

### 3. check_docs_links.py (Original)

**Purpose:** Legacy link checker (simple version, kept for compatibility).

**Usage:**
```bash
python scripts/check_docs_links.py
```

**Note:** Use the enhanced version above for more comprehensive checks.

---

## Configuration Files

### .markdownlint.json

**Purpose:** Configuration for markdownlint (markdown linter).

**Location:** Project root (`.markdownlint.json`)

**Key rules:**
- `MD001`: Heading levels must increment by one (H1 → H2 → H3)
- `MD003`: Use ATX-style headings (`#` not `===`)
- `MD004`: Use dash (`-`) for unordered lists
- `MD013`: Line length limit (disabled for flexibility)
- `MD025`: Only one H1 per document

**Modify rules:**
Edit `.markdownlint.json` to enable/disable rules.

**Example (disable MD013 for one file):**
```markdown
<!-- markdownlint-disable MD013 -->
This is a really long line that exceeds 80 characters but that's OK here.
<!-- markdownlint-enable MD013 -->
```

---

## Installation

### Python Dependencies (Required)

```bash
pip install mkdocs mkdocs-material mkdocstrings
```

### Node.js Dependencies (Optional)

```bash
# Markdownlint (optional but recommended)
npm install -g markdownlint-cli
```

**Note:** `validate_docs.py` works without markdownlint, but will skip that check.

---

## Usage in CI/CD

### GitHub Actions

**Example workflow (`.github/workflows/docs.yml`):**

```yaml
name: Documentation Validation

on:
  pull_request:
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
      - '.markdownlint.json'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install mkdocs mkdocs-material mkdocstrings
          npm install -g markdownlint-cli
      
      - name: Validate documentation
        run: python scripts/validate_docs.py --full
      
      - name: Build docs
        run: mkdocs build --strict
```

---

## Pre-Commit Hook (Recommended)

**Setup:**

1. Create `.git/hooks/pre-commit`:
   ```bash
   #!/bin/bash
   # Run documentation validation before committing
   
   # Check if docs/ modified
   git diff --cached --name-only | grep -q '^docs/'
   if [ $? -eq 0 ]; then
       echo "Validating documentation..."
       python scripts/validate_docs.py --skip-build
       if [ $? -ne 0 ]; then
           echo "❌ Documentation validation failed. Fix errors before committing."
           exit 1
       fi
   fi
   ```

2. Make executable:
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

**Effect:** Automatically validates docs before every commit.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'mkdocs'"

**Fix:**
```bash
pip install mkdocs mkdocs-material mkdocstrings
```

---

### "markdownlint: command not found"

**Fix (optional):**
```bash
npm install -g markdownlint-cli
```

**Or skip:**
Validation works without markdownlint, but will skip that check.

---

### "Permission denied" when running scripts

**Fix:**
```bash
chmod +x scripts/check_docs_links.py
chmod +x scripts/validate_docs.py
```

---

### Validation passes locally but fails in CI

**Common causes:**
1. **Uncommitted files:** Images/files not added to git
2. **Case-sensitive paths:** Linux is case-sensitive (use lowercase filenames)
3. **Line endings:** Use LF, not CRLF (Windows issue)

**Fix:**
```bash
# Check for uncommitted files
git status

# Fix line endings (Windows)
git config core.autocrlf input
```

---

## Related Documentation

- **[Documentation Style Guide](../docs/06-developer-guide/documentation_style_guide.md)** — Writing guidelines
- **[Documentation Maintainer Guide](../docs/06-developer-guide/documentation_maintainer_guide.md)** — Maintainer workflow
- **[Documentation Guidelines](../docs/06-developer-guide/documentation_guidelines.md)** — High-level guidelines

---

## Support

**Questions or issues with validation scripts?**
- **GitHub Issues:** [foodspec/issues](https://github.com/chandrasekarnarayana/foodspec/issues) (label: `tooling`)
- **GitHub Discussions:** [foodspec/discussions](https://github.com/chandrasekarnarayana/foodspec/discussions)

---

**Last updated:** December 28, 2024
