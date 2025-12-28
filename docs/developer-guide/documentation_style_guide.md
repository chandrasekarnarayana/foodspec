# Documentation Style Guide

**Purpose:** Maintain consistent, high-quality documentation across the FoodSpec project.

**Audience:** Contributors writing or updating documentation (tutorials, workflows, API docs, etc.).

---

## Quick Checklist (Definition of Done)

Before submitting any new documentation page, verify:

- [ ] Headings follow hierarchy (H1 → H2 → H3, no skipping)
- [ ] All code blocks have language tags (```python, ```bash, etc.)
- [ ] All links are relative (no absolute URLs to internal docs)
- [ ] All images have descriptive alt text
- [ ] At least one working code example (if applicable)
- [ ] Cross-references to related pages
- [ ] Runs `mkdocs build` without errors
- [ ] Passes `scripts/check_docs_links.py` (no broken links)
- [ ] Passes `markdownlint` (if configured)
- [ ] Added to `mkdocs.yml` navigation
- [ ] Reviewed for spelling/grammar (use spell checker)

---

## 1. Writing Style & Tone

### General Principles

**✅ DO:**
- Write in **active voice**: "Run the command" (not "The command should be run")
- Be **concise**: Remove filler words ("just", "simply", "basically")
- Use **present tense**: "FoodSpec processes spectra" (not "will process")
- Address the reader as **"you"**: "You can use..." (not "One can use...")
- Explain **why**, not just **what**: "Use ALS baseline correction because it preserves sharp peaks"

**❌ DON'T:**
- Use jargon without defining it (define technical terms on first use)
- Write walls of text (break into sections, lists, or tables)
- Assume prior knowledge (link to background docs for complex concepts)
- Use emojis excessively (sparingly OK for emphasis: ✅ ❌ 🚀)

### Example: Before/After

**❌ BAD (Passive, wordy):**
> "The baseline correction functionality can be applied to the spectra by using the baseline_als() function, which should be called with appropriate parameters."

**✅ GOOD (Active, concise):**
> "Correct the baseline using `baseline_als()`. Set `lam=1e6` for FTIR or `lam=1e5` for Raman."

---

## 2. Document Structure

### 2.1 Headings

**Rules:**
1. **One H1 per page** (`# Title`) — matches the page title
2. **Hierarchical nesting:** H1 → H2 → H3 (never skip levels: H1 → H3 ❌)
3. **Sentence case:** "How to preprocess spectra" (not "How To Preprocess Spectra")
4. **No trailing periods:** `## Installation` (not `## Installation.`)
5. **Descriptive, not generic:** `## Load FTIR Spectra from CSV` (not `## Step 1`)

**Example Hierarchy:**
```markdown
# Oil Authentication Workflow  ← H1 (once per page)

## Overview  ← H2
Brief description...

## Step 1: Load Data  ← H2
Instructions...

### CSV Format  ← H3 (nested under H2)
Details...

### HDF5 Format  ← H3
Details...

## Step 2: Preprocess  ← H2 (back to H2 level)
...
```

---

### 2.2 Page Template

**Every tutorial/workflow should follow this structure:**

```markdown
# [Title]

**Purpose:** One-sentence summary (e.g., "Authenticate olive oils using FTIR spectroscopy").

**Audience:** Who is this for? (e.g., "Food scientists with basic Python knowledge").

**Time:** Estimated completion time (e.g., "20 minutes").

**Prerequisites:** Required knowledge/tools (e.g., "Install FoodSpec, have CSV data").

---

## Overview
- What you'll learn
- What problem this solves
- Key concepts

## Step 1: [Action]
Instructions...

## Step 2: [Action]
Instructions...

## Complete Code
Full working example (copy-paste ready)

## Expected Output
What the user should see

## Troubleshooting
Common issues and fixes

## Next Steps
Links to related tutorials/workflows

---

**Related Pages:**
- Link 1: path/to/page.md
- Link 2: path/to/page.md
```

---

<a id="code-blocks"></a>
## 3. Code Blocks

### 3.1 Language Tags (REQUIRED)

**Always specify the language:**

✅ **GOOD:**
````markdown
```python
from foodspec import SpectralDataset
```
````

❌ **BAD (missing language tag):**
````markdown
```python
from foodspec import SpectralDataset
```
````

**Common language tags:**
- `python` — Python code
- `bash` — Shell commands (Linux/macOS)
- `console` — Terminal output (use `$` prefix for commands)
- `yaml` — YAML configuration files
- `json` — JSON data
- `text` — Plain text (logs, output)
- `markdown` — Markdown examples

---

### 3.2 Python Code Rules

**1. Imports at the top:**
```python
# ✅ GOOD
from foodspec import SpectralDataset
from sklearn.ensemble import RandomForestClassifier
import numpy as np

ds = SpectralDataset.from_csv('data.csv')
```

```python
# ❌ BAD (imports scattered)
from foodspec import SpectralDataset
ds = SpectralDataset.from_csv('data.csv')
from sklearn.ensemble import RandomForestClassifier  # ← Import later ❌
```

**2. Use print() for outputs:**
```python
# ✅ GOOD (shows what user will see)
accuracy = 0.95
print(f"Accuracy: {accuracy:.3f}")
# Output: Accuracy: 0.950
```

**3. Include comments for non-obvious steps:**
```python
# ✅ GOOD
# ALS removes baseline drift; lambda=1e6 for smooth baseline
X_corrected = baseline_als(X, lam=1e6, p=0.01)
```

**4. Keep code blocks short (<30 lines):**
- If longer, break into multiple blocks with explanatory text between them
- Or provide a "Complete Code" section at the end

---

### 3.3 Shell Command Rules

**Use `$` prefix for commands, no prefix for output:**

```bash
# ✅ GOOD
$ foodspec --version
FoodSpec 1.0.0

$ ls data/
oils.csv  spectra.h5
```

**For multi-line commands, use backslash:**
```bash
$ foodspec-run-protocol \
  --input oils.csv \
  --protocol oil_authentication \
  --output-dir runs/demo
```

---

### 3.4 Expected Output

**Always show what the user should expect:**

```python
from foodspec import SpectralDataset

ds = SpectralDataset.from_csv('oils.csv')
print(f"Loaded {len(ds)} spectra")
# Expected output:
# Loaded 30 spectra
```

---

## 4. Links

### 4.1 Internal Links (Relative Paths)

**✅ ALWAYS use relative paths:**
```markdown
[Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md)
```

**❌ NEVER use absolute URLs:**
```markdown
<!-- ❌ BAD -->
[Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md)
```

**Path Rules:**
- From `docs/tutorials/beginner/01-load-and-plot.md` to `docs/methods/preprocessing/normalization_smoothing.md`:
  ```markdown
    [Preprocessing](../methods/preprocessing/normalization_smoothing.md)
  ```
- Same directory: FAQ (example)
- Parent directory: `[Home](../index.md)`
- Child directory: tutorials/oil_auth.md (example)

---

### 4.2 Anchor Links

**Link to specific sections using `#heading-slug`:**

```markdown
See [Installation → Python Requirements](../getting-started/installation.md#python-requirements)
```

**Heading slug rules:**
- Lowercase
- Replace spaces with hyphens
- Remove special characters
- Example: `## Python 3.8+ Requirements` → `#python-38-requirements`

---

### 4.3 External Links

**Format:**
```markdown
[Link text](https://example.com)
```

**For long URLs, use reference-style:**
```markdown
See the [NumPy documentation][numpy-docs] for details.

[numpy-docs]: https://numpy.org/doc/stable/
```

---

### 4.4 Code References

**Link to specific functions in API docs:**

```markdown
Use [`baseline_als()`](../api/preprocessing.md#baseline_als) to correct the baseline.
```

**Inline code (backticks):**
```markdown
The `SpectralDataset` class provides...
```

---

## 5. Images & Figures

### 5.1 Naming Convention

**Format:** `<section>_<description>_<optional-detail>.png`

**Examples:**
- `preprocessing_baseline_comparison.png` — Shows before/after baseline correction
- `workflow_oil_auth_pipeline.png` — Oil authentication workflow diagram
- `tutorial_pca_scores_plot.png` — PCA scores plot from tutorial

**Rules:**
- Lowercase with underscores
- Descriptive (not `image1.png` ❌)
- Use `.png` for screenshots/plots, `.svg` for diagrams (vector graphics)

---

### 5.2 Storage Location

**Store images in:**
```plaintext
docs/assets/images/<section>/
```

**Examples:**
- `docs/assets/images/tutorials/oil_auth_result.png`
- `docs/assets/images/workflows/heating_quality_flowchart.svg`
- `docs/assets/images/preprocessing/baseline_als_demo.png`

---

### 5.3 Alt Text (REQUIRED for Accessibility)

**✅ GOOD (descriptive alt text):**
```markdown
![Bar chart showing 95% classification accuracy for olive oil authentication](../../assets/figures/oil_confusion.png)
```

**❌ BAD (generic alt text - example only):**
```markdown
<!-- Example of bad alt text - DO NOT USE; placeholder commented to avoid broken link checks -->
<!-- ![image](../../assets/pca_scores.png) -->
```

**Alt text rules:**
- Describe what the image **shows**, not what it **is**
- ~10-20 words
- No need for "Image of..." (screen readers add this)

---

### 5.4 Image Size

**Guidelines:**
- **Screenshots:** 800-1200px wide (readable but not huge)
- **Plots:** 600-800px wide (matplotlib: `figsize=(8, 6)`)
- **Diagrams:** SVG (scalable) or PNG at 1000px wide

**Optimize file size:**
```bash
# Install optimizer
$ pip install pillow

# Optimize PNG (reduce file size without quality loss)
$ python -m PIL.Image input.png --optimize --quality=85 output.png
```

---

### 5.5 Captions

**Use italics for captions:**
```markdown
![PCA scores plot showing class separation](../../assets/pca_scores.png)

*Figure 1: PCA scores plot shows clear separation between EVOO (blue) and lampante (red) olive oils.*
```

*(Note: Example images in this style guide are for illustration only and may not exist)*

---

## 6. Lists & Tables

### 6.1 Unordered Lists

**Use `-` (hyphen) consistently:**

```markdown
- Item 1
- Item 2
  - Sub-item 2.1 (indent with 2 spaces)
  - Sub-item 2.2
- Item 3
```

**Not `*` or `+` (inconsistent with existing docs).**

---

### 6.2 Ordered Lists

**Use `1.` for all items (auto-numbered):**

```markdown
1. First step
1. Second step
1. Third step
```

(Markdown auto-increments: 1, 2, 3...)

---

### 6.3 Tables

**Always include header row:**

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value A  | Value B  | Value C  |
| Value D  | Value E  | Value F  |
```

**Alignment:**
- Left: `|----------|`
- Center: `|:--------:|`
- Right: `|---------:|`

**Example:**
```markdown
| Method | Accuracy | Speed |
|--------|:--------:|------:|
| ALS    | 95%      | 12 ms |
| Poly   | 92%      | 3 ms  |
```

---

## 7. Admonitions (Callouts)

**Use Material for MkDocs admonitions for emphasis:**

```markdown
!!! note "Optional Title"
    This is a note. Use for supplementary information.

!!! warning
    This is a warning. Use for potential pitfalls.

!!! tip
    This is a tip. Use for helpful advice.

!!! danger
    This is a danger box. Use for critical warnings (data loss, etc.).

!!! info
    This is an info box. Use for general information.

!!! example
    This is an example box. Use for worked examples.

!!! quote
    This is a quote box. Use for citations or quotes.
```

**Example Usage:**

```markdown
!!! warning "Data Leakage Risk"
    Always use `GroupKFold` to prevent replicate leakage. See [Validation Guide](../methods/validation/cross_validation_and_leakage.md).

!!! tip
    Use `baseline_als(lam=1e6)` for FTIR and `lam=1e5` for Raman spectra.
```

---

## 8. How to Add a New Workflow

**Step-by-step process:**

### 1. Create the markdown file

**Location:** `docs/workflows/<workflow_name>.md`

**Example:** `docs/workflows/olive_oil_authentication.md`

**Template:**
```markdown
# [Workflow Name]

**Purpose:** One-sentence summary.

**Use case:** Who needs this? What problem does it solve?

**Time:** 20-30 minutes

**Prerequisites:**
- FoodSpec installed
- Basic Python knowledge
- CSV data with labels

---

## Overview
Brief description of the workflow...

## Step 1: Load and Inspect Data
...

## Step 2: Preprocess
...

## Step 3: Train Model
...

## Step 4: Validate
...

## Complete Code
Full working example...

## Expected Output
What the user should see...

## Troubleshooting
Common issues...

## Next Steps
- Related Workflow 1 (example)
- Related Workflow 2 (example)

---

**Related Pages:**
- [Preprocessing Guide](../methods/preprocessing/normalization_smoothing.md)
- [Cross-Validation & Leakage](../methods/validation/cross_validation_and_leakage.md)
```

---

### 2. Add to mkdocs.yml navigation

**Open:** `mkdocs.yml`

**Add under `Applications & Workflows` section:**

```yaml
  # Level 6: Applications & Workflows (Domain Expertise)
  - Applications & Workflows:
      - Workflows Overview: workflows/index.md
      - Authentication & Identification:
          - Oil Authentication: workflows/oil_authentication.md
          - [NEW WORKFLOW]: workflows/authentication/oil_authentication.md  # ← Add here
```

---

### 3. Add to workflows index

**Open:** `docs/workflows/index.md`

**Add to the workflow list:**

```markdown
## Available Workflows

| Workflow | Use Case | Difficulty | Time |
|----------|----------|------------|------|
| Olive Oil Authentication (authentication/oil_authentication.md) | Classify oil varieties | Beginner | 20 min |
```

---

### 4. Test locally

```bash
# Build docs
$ mkdocs build

# Check for broken links
$ python scripts/check_docs_links.py

# Serve locally (preview at http://localhost:8000)
$ mkdocs serve
```

---

### 5. Create example data (if needed)

**If your workflow needs sample data:**

1. Create CSV in `examples/data/<workflow_name>_demo.csv`
2. Document the format in the workflow page
3. Provide download link or generation script

**Example:**
```python
# Generate demo data
import numpy as np
import pandas as pd

np.random.seed(42)
wavenumbers = np.linspace(1000, 3000, 150)
X = np.random.normal(5, 0.3, (30, 150))
y = ['EVOO'] * 15 + ['Lampante'] * 15

df = pd.DataFrame(X, columns=[f'{w:.1f}' for w in wavenumbers])
df.insert(0, 'label', y)
df.to_csv('examples/data/olive_oil_demo.csv', index=False)
```

---

## 9. How to Add a New Tutorial

**Similar to workflows, but more educational (step-by-step learning).**

### 1. Choose the difficulty level

- **Level 1 (Beginner):** 5-15 minutes, single concept
- **Level 2 (Applied):** 20-40 minutes, real-world example
- **Level 3 (Advanced):** 45-90 minutes, complex integration

### 2. Create the file

**Location:** `docs/tutorials/<category>/<name>.md`

**Example:** `docs/tutorials/beginner/01-load-and-plot.md`

---

### 3. Follow the tutorial template

```markdown
# [Tutorial Name]

**Difficulty:** Level 1 (Beginner) / Level 2 (Applied) / Level 3 (Advanced)

**Time:** 10 minutes

**Learning objectives:**
- Learn X
- Understand Y
- Practice Z

**Prerequisites:**
- FoodSpec installed
- (Optional) Jupyter notebook

---

## Background
Brief theory (1-2 paragraphs)...

## Hands-On Exercise

### Step 1: [Action]
Instructions...

```
# Code here
```plaintext

**Expected output:**
```
Output here
```yaml

### Step 2: [Action]
...

## Summary
What you learned...

## Practice Exercises
1. Try modifying X to see Y
2. Apply this to your own data

## Next Tutorial
[Next Level Tutorial](../tutorials/intermediate/01-oil-authentication.md)

---

**Related Pages:**
- [Theory](../theory/spectroscopy_basics.md)
- [API Reference](../api/index.md)
```

---

### 4. Add to mkdocs.yml

**Add under appropriate level:**

```yaml
  # Level 7: Tutorials & Learning (Step-by-Step)
  - Tutorials:
      - Tutorial Ladder: 02-tutorials/index.md
      - Level 1 - Beginner (5-15 min):
          - Baseline Correction: 02-tutorials/level1_baseline_and_smoothing.md  # ← Add here
```yaml

---

### 5. Update tutorial index

**Open:** `docs/02-tutorials/index.md`

**Add to the ladder:**

```
## Level 1: Beginner (5–15 min)

| Tutorial | What You'll Learn | Time |
|----------|-------------------|------|
| [Baseline Correction](../tutorials/beginner/01-load-and-plot.md) | Remove baseline drift using ALS | 10 min |
```yaml

---

## 10. Definition of Done (Checklist)

**Before submitting a PR with new documentation:**

### Content Quality
- [ ] Headings follow hierarchy (H1 → H2 → H3)
- [ ] Active voice, present tense, concise
- [ ] At least one working code example
- [ ] Code blocks have language tags (```python, etc.)
- [ ] All outputs shown (print statements, expected results)
- [ ] Technical terms defined or linked to glossary
- [ ] Cross-references to 2+ related pages

### Links & Images
- [ ] All internal links are relative paths
- [ ] All images have descriptive alt text
- [ ] Image file names follow convention (`section_description.png`)
- [ ] No broken links (run `scripts/check_docs_links.py`)

### Navigation & Indexing
- [ ] Added to `mkdocs.yml` navigation
- [ ] Added to section index page (e.g., `workflows/index.md`)
- [ ] Listed in "Related Pages" sections of relevant docs

### Testing
- [ ] `mkdocs build` succeeds (no errors)
- [ ] `mkdocs serve` renders correctly (check locally)
- [ ] `scripts/check_docs_links.py` passes
- [ ] Code examples tested (copy-paste and run)
- [ ] Spell-checked (VS Code spell checker or similar)

### Metadata & Cleanup
- [ ] File name is lowercase with underscores
- [ ] No trailing whitespace (use auto-formatter)
- [ ] Consistent indentation (2 spaces for YAML, 4 for Python)
- [ ] Git commit message descriptive (`docs: Add olive oil authentication workflow`)

---

## 11. Common Pitfalls

### ❌ Pitfall 1: Skipping heading levels

```
# Main Title (H1)
### Subsection (H3) ← ❌ Skipped H2!
```yaml

**Fix:** Always go H1 → H2 → H3.

---

### ❌ Pitfall 2: Absolute URLs to internal docs

```
[Guide](https://chandrasekarnarayana.github.io/foodspec/guide/) ← ❌
```yaml

**Fix:** Use relative paths: `[Guide](documentation_guidelines.md)`

---

### ❌ Pitfall 3: Missing language tags

````markdown
```yaml
from foodspec import SpectralDataset  ← ❌ No language tag
```
````

**Fix:** Add `python` tag.

---

### ❌ Pitfall 4: Untested code examples

```python
# ❌ This code doesn't actually work!
ds = SpectralDataset.from_csv('nonexistent_file.csv')
```

**Fix:** Test all code examples before publishing.

---

### ❌ Pitfall 5: No expected output

```python
print(f"Accuracy: {accuracy:.3f}")
# ❌ User doesn't know what to expect
```

**Fix:** Add comment:
```python
print(f"Accuracy: {accuracy:.3f}")
# Expected output: Accuracy: 0.950
```

---

## 12. Documentation Linting & Validation

**Automated checks to maintain quality.**

### 12.1 Markdownlint

**Install:**
```bash
$ npm install -g markdownlint-cli
```

**Run:**
```bash
$ markdownlint docs/**/*.md
```

**Configuration:** See `.markdownlint.json` (if configured).

---

### 12.2 Link Checker

**Script:** `scripts/check_docs_links.py`

**Run:**
```bash
$ python scripts/check_docs_links.py
```

**What it checks:**
- Broken internal links (missing files)
- Invalid anchor links (non-existent headings)
- Broken external URLs (optional, slower)

---

### 12.3 MkDocs Build

**Always run before submitting:**

```bash
$ mkdocs build
```

**Look for:**
- `ERROR` lines (broken links, missing files)
- `WARNING` lines (investigate and fix if critical)

---

## 13. Maintainer Instructions

**For maintainers reviewing documentation PRs:**

### Pre-Merge Checklist

1. **Automated checks pass:**
   ```bash
   $ mkdocs build  # No errors
   $ python scripts/check_docs_links.py  # No broken links
   $ markdownlint docs/**/*.md  # (If configured)
   ```

2. **Manual review:**
   - [ ] Code examples tested (copy-paste and run)
   - [ ] Screenshots/images are clear and appropriately sized
   - [ ] Tone is consistent with existing docs
   - [ ] No duplicate content (link to existing docs instead)

3. **Test locally:**
   ```bash
   $ mkdocs serve
   # Open http://localhost:8000
   # Navigate to the new page
   # Check rendering, links, images
   ```

4. **Check mobile rendering:**
   - Resize browser to mobile width
   - Verify tables don't overflow
   - Ensure images scale properly

---

### Post-Merge Tasks

1. **Verify deployment:**
   - Wait for GitHub Actions to deploy
   - Check live site: https://chandrasekarnarayana.github.io/foodspec/

2. **Update index pages:**
   - If new tutorial → update `docs/02-tutorials/index.md`
   - If new workflow → update `docs/workflows/index.md`

3. **Announce in release notes:**
   - Add to `CHANGELOG.md` under "Documentation"
   - Mention in next release notes

---

## 14. Tools & Resources

### Recommended Tools

| Tool | Purpose | Link |
|------|---------|------|
| **VS Code** | Markdown editor with preview | [code.visualstudio.com](https://code.visualstudio.com/) |
| **Markdown All in One** | VS Code extension (auto-formatting, TOC) | [Marketplace](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) |
| **markdownlint** | Markdown linter (CLI or VS Code extension) | [GitHub](https://github.com/DavidAnson/markdownlint) |
| **MkDocs Material** | Theme documentation | [squidfunk.github.io/mkdocs-material](https://squidfunk.github.io/mkdocs-material/) |
| **Grammarly** | Spell/grammar checker | [grammarly.com](https://www.grammarly.com/) |

---

### Style References

- **Microsoft Writing Style Guide:** [docs.microsoft.com/style-guide](https://docs.microsoft.com/en-us/style-guide/)
- **Google Developer Documentation Style Guide:** [developers.google.com/style](https://developers.google.com/style)
- **Markdown Guide:** [markdownguide.org](https://www.markdownguide.org/)

---

## 15. Examples of Good Documentation

**Internal examples (FoodSpec):**
- [Oil Authentication Workflow](../workflows/authentication/oil_authentication.md) — Clear structure, working code
- [Troubleshooting Guide](../10-help/troubleshooting.md) — Problem → Solution format
- [Validation & Leakage](../methods/validation/cross_validation_and_leakage.md) — Theory + practice

**External examples (inspiration):**
- **scikit-learn:** [scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- **FastAPI:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com/)
- **NumPy:** [numpy.org/doc/stable/user/](https://numpy.org/doc/stable/user/)

---

## 16. FAQ (Meta)

**Q: How long should a tutorial be?**  
A: Level 1: 5-15 min, Level 2: 20-40 min, Level 3: 45-90 min. Break long tutorials into multiple parts.

**Q: Should I include theory or just code?**  
A: Balance both. Start with brief theory (1-2 paragraphs), then code. Link to detailed theory in "Theory & Background" section.

**Q: How many code examples per page?**  
A: At least 1-2 working examples. More is better, but keep each block <30 lines.

**Q: Can I reuse content from other pages?**  
A: No. Link to the existing page instead. Reduces maintenance burden.

**Q: What if I find outdated documentation?**  
A: Open a GitHub issue or submit a PR to update it.

---

## 17. Contact & Questions

**Need help with documentation?**
- **Ask in GitHub Discussions:** [foodspec/discussions](https://github.com/chandrasekarnarayana/foodspec/discussions)
- **Open an issue:** [foodspec/issues](https://github.com/chandrasekarnarayana/foodspec/issues) (label: `documentation`)
- **Tag maintainers:** @chandrasekarnarayana (for urgent issues)

---

**Last updated:** December 28, 2024 | **Version:** 1.0.0
