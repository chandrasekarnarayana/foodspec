# FoodSpec Documentation Guidelines

**Version:** 1.0  
**Last Updated:** 2025-12-25  
**Status:** CANONICAL RULEBOOK

This document defines the mandatory rules for all FoodSpec documentation. Every page, tutorial, and guide must comply with these standards. Non-compliance blocks publication.

---

## RULE 1: Question-First Context Block (Mandatory)

Every documentation page MUST start with a context block answering:

```markdown
<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** [Target audience: layman/food scientist/spectroscopist/physicist/data scientist/engineer/reviewer]  
**What problem does this solve?** [Plain English problem statement]  
**When to use this?** [Conditions/scenarios where this applies]  
**Why it matters?** [Impact/consequences of using or not using this]  
**Time to complete:** [Estimated reading/execution time: 5 min / 15 min / 1 hour]  
**Prerequisites:** [What you must know/have before starting]
```

**Example:**
```markdown
**Who needs this?** Food scientists authenticating edible oils in QA labs.  
**What problem does this solve?** Detecting olive oil adulteration using Raman spectroscopy.  
**When to use this?** When you have Raman spectra and need to verify oil authenticity.  
**Why it matters?** Adulterated oils violate standards and harm consumer trust.  
**Time to complete:** 15 minutes  
**Prerequisites:** FoodSpec installed; CSV file with Raman spectra; basic terminal knowledge
```

---

## RULE 2: Multi-Audience Layering (4 Layers)

Every core concept MUST be explained in 4 progressive layers:

### Layer 1: Layman (Plain English)
- No jargon, no equations
- Everyday analogies
- "What is it?" in 2-3 sentences

**Example:**  
"Raman spectroscopy is like shining a flashlight on food and measuring the colors that bounce back. Different molecules reflect different colors, so we can identify what's in the sample."

### Layer 2: Domain Expert (Food Scientist / Spectroscopist)
- Domain-specific terminology
- Practical applications
- "How does this apply to my work?"

**Example:**  
"Raman spectroscopy measures molecular vibrations induced by inelastic scattering. For edible oils, we analyze carbonyl/unsaturation bands (1600-1800 cm⁻¹) to discriminate between olive, sunflower, and palm oils."

### Layer 3: Rigorous Theory (Physicist / Chemometrician)
- Mathematical formulations
- Assumptions and derivations
- Validation requirements
- Failure modes

**Example:**  
"Raman intensity $I(\nu)$ is proportional to the scattering cross-section $\frac{d\sigma}{d\Omega}$ and molecular polarizability derivative $\frac{\partial\alpha}{\partial Q}$. For quantitative analysis, assume: (1) linear detector response, (2) negligible fluorescence, (3) stable laser power. **When results cannot be trusted:** strong fluorescence interference, low signal-to-noise ratio (SNR < 10), or thermal degradation during measurement."

### Layer 4: Developer / API
- Code examples
- Function signatures
- Integration patterns

**Example:**  
```python
from foodspec import SpectralDataset
from foodspec.apps.oils import run_oil_authentication_workflow

# Load Raman spectra
ds = SpectralDataset.from_csv("oils.csv", wavenumber_col="wavenumber", intensity_cols="auto")

# Run authentication
results = run_oil_authentication_workflow(ds, validation_mode="batch_aware")
```

---

## RULE 3: Progressive Disclosure Structure

Every page MUST follow this order:

1. **Context Block** (RULE 1)
2. **What?** (Layer 1-2 explanation)
3. **Why?** (Scientific rationale, applications)
4. **When?** (Use cases, decision criteria)
5. **How?** (Step-by-step instructions, code examples)
6. **Pitfalls & Limitations** (Common mistakes, failure modes, "When results cannot be trusted")
7. **What's Next?** (Cross-links to related pages)

**No reordering allowed.** Users must encounter content in this sequence.

---

## RULE 4: Canonical Home Rule

Every feature/concept has ONE canonical home. Other pages link to it.

**Examples:**
- RQ engine theory → Canonical home: `docs/07-theory-and-background/rq_engine_theory.md`
- Preprocessing → Canonical home: `docs/04-user-guide/preprocessing_guide.md`
- Protocol YAML schema → Canonical home: `docs/04-user-guide/protocols_and_yaml.md`

**Enforcement:**
- No duplicated full explanations
- If a concept appears elsewhere, link to canonical home with 1-sentence summary
- Update canonical home, not scattered mentions

**How to check:**
```bash
rg -n "PCA|cross-validation|RQ engine" docs --type md | sort
# If a term appears in >3 files with >5 lines each, consolidate to canonical home
```

---

## RULE 5: Validation + Limits (Mandatory for All Algorithms)

Every algorithm/workflow page MUST include:

### 5.1 Assumptions
List all assumptions required for correct operation.

**Example:**
- Input spectra are baseline-corrected
- Wavenumber range includes 1600-1800 cm⁻¹
- At least 3 replicates per sample

### 5.2 Data Requirements
Specify minimum dataset requirements.

**Example:**
- Minimum 30 spectra per class for classification
- At least 2 batches for batch-aware cross-validation
- No missing values in target columns

### 5.3 Validation Strategy
Explain how to verify results.

**Example:**
- Balanced accuracy >0.7 indicates acceptable performance
- Check confusion matrix for systematic misclassifications
- Verify feature importance scores are reproducible across CV folds

### 5.4 Failure Modes & When Results Cannot Be Trusted

**THIS SECTION IS MANDATORY.** Every theory/workflow page must have a clearly marked section:

```markdown
## When Results Cannot Be Trusted

Results are unreliable when:
1. [Condition 1 with detection method]
2. [Condition 2 with detection method]
3. [Condition 3 with detection method]

**How to detect:**
- [Diagnostic metric/plot to check]

**What to do:**
- [Remediation steps or warnings to report]
```

**Example:**
```markdown
## When Results Cannot Be Trusted

Results are unreliable when:
1. **Class imbalance >10:1** – Balanced accuracy becomes optimistic; check per-class F1 scores.
2. **CV fold variance >0.2** – Unstable model; increase sample size or reduce feature count.
3. **Test accuracy >> training accuracy** – Data leakage; verify batch-aware splitting.

**How to detect:**
- Check `results['cv_metrics']['fold_std']` in output JSON
- Inspect confusion matrix for empty rows/columns

**What to do:**
- Use stratified sampling to balance classes
- Run 10-fold CV instead of 5-fold to reduce variance
- Review preprocessing for leakage (e.g., global normalization before splitting)
```

---

## RULE 6: Runnable + Tested Examples

All code examples MUST be:
- **Runnable as-is** (no placeholders like `<your_file.csv>`)
- **Tested** in CI or manually verified before each release
- **Match current API** (no deprecated function calls)

**Verification:**
```bash
# Extract all code blocks from docs
rg -A 20 '```python' docs --type md > /tmp/doc_examples.py
# Run syntax check
python -m py_compile /tmp/doc_examples.py
```

**Enforcement:**
- Every release checklist includes: "Run all documentation code examples"
- Use `examples/` directory for canonical data files referenced in docs

---

## PAGE METADATA (TITLES, DESCRIPTIONS, SOCIAL)

- Use a clear, reader-focused H1 title that matches the page’s purpose.
- Add a concise first paragraph (1–2 sentences) that doubles as the default meta description.
- Optional front matter keys (only when needed):
  - `title`: Override the page title (rarely needed).
  - `description`: Custom meta description; otherwise rely on the first paragraph.
  - `canonical_url`: For mirrored content only.
  - `image`: Social preview image (1200x630 preferred). Store under `docs/assets/` or a CDN. Reference it in front matter when a page needs a specific preview.

**Social preview guidance:**
- Prefer a single default preview image under `docs/assets/` for consistency.
- For marquee pages (e.g., flagship tutorials, release notes), set `image` in front matter and ensure the asset exists and has alt text when embedded.

---

## RULE 7: No Orphan Pages

Every page in `docs/` MUST either:
1. Appear in `mkdocs.yml` navigation, OR
2. Live in `docs/_internal/archive/` with an ARCHIVED banner

**ARCHIVED banner template:**
```markdown
---
!!! warning "ARCHIVED"
    This page is preserved for historical reference but is no longer maintained.  
    See [index.md](../index.md) for current documentation.
---
```

**Verification:**
```bash
# Find markdown files not in mkdocs.yml and not in _internal/
comm -23 \
  <(find docs -name "*.md" ! -path "*/_internal/*" | sort) \
  <(rg "^\s+- .*\.md" mkdocs.yml | sed 's/.*: //' | sort)
```

---

## RULE 8: Versioning Notes Standard

When documenting version-specific behavior:

```markdown
> **Version Note:**  
> - v0.1.x: Old behavior [deprecated]  
> - v0.2.0+: New behavior [recommended]  
> - Breaking changes: [What changed and how to migrate]
```

**Example:**
```markdown
> **Version Note:**  
> - v0.1.x: `foodspec-run-protocol` required `--input-csv` flag  
> - v0.2.0+: Use `--input` (supports CSV/HDF5)  
> - Breaking changes: Old `--input-csv` flag removed; use `--input` instead
```

---

## RULE 9: Cross-Linking Standard

Use relative links with descriptive text:

**Good:**
```markdown
See [RQ engine theory](../theory/rq_engine_detailed.md) for mathematical foundations.
```

**Bad:**
```markdown
See [here](../theory/rq_engine_detailed.md).
Click [this link](../theory/rq_engine_detailed.md).
```

**Enforcement:**
- No bare URLs
- No "click here" / "this page" link text
- Links must survive directory restructuring (use relative paths)

---

## RULE 10: Archive Management

When archiving a page:

1. Move file to `docs/_internal/archive/`
2. Add ARCHIVED banner (RULE 7)
3. Remove from `mkdocs.yml` navigation
4. Add redirect comment at old location if moved

**Archive landing page:**  
`docs/_internal/archive/README.md` lists all archived pages with dates and reasons.

---

## Enforcement Checklist

Before merging any documentation change:

- [ ] Context block present (RULE 1)
- [ ] 4-layer explanation for core concepts (RULE 2)
- [ ] Progressive disclosure structure followed (RULE 3)
- [ ] No duplicated explanations; canonical home linked (RULE 4)
- [ ] "When Results Cannot Be Trusted" section present (RULE 5)
- [ ] Code examples are runnable and tested (RULE 6)
- [ ] Page appears in mkdocs.yml or has ARCHIVED banner (RULE 7)
- [ ] Version notes use standard template (RULE 8)
- [ ] Cross-links use descriptive text (RULE 9)
- [ ] Archived pages moved correctly (RULE 10)

---

## Contact

Questions about these guidelines? Open an issue or discussion at:  
https://github.com/chandrasekarnarayana/foodspec/discussions
