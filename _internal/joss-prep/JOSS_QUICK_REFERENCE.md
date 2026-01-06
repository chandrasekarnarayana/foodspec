# FoodSpec JOSS Documentation: Quick Reference

## 📋 One-Page Summary

**Goal:** Upgrade 10 key documentation pages to JOSS standards in 2-3 hours  
**Format:** Page Contract template (Purpose, Audience, Time, Prerequisites, Runnable Example, Next Steps)  
**Critical Path:** 2 pages (quickstart, CLI guide) = 45 minutes  
**Full Upgrade:** 10 pages = 2 hours 45 minutes  

---

## ⚡ 30-Second Page Contract Structure

```markdown
# Title (H1, one per page)

**Purpose:** [What will I learn?]
**Audience:** [Who is this for?]
**Time:** [5-15 minutes]
**Prerequisites:** [What I need first]

---

## [Action Name]
[Code + expected output]

---

## Next Steps
1. [Link](path.md) — Why link
2. [Link](path.md) — Why link
3. [Link](path.md) — Why link
```

---

## 🎯 Priority Pages (Fast Track)

### BLOCKERS (Do First: 45 min)
| # | Page | What's Missing | Time |
|---|------|--------|------|
| 1 | `quickstart_15min.md` | Python code example | 15 min |
| 2 | `first-steps_cli.md` | Context block at top | 20 min |

### MUST-HAVES (Do Second: 2 hours)
| 3 | `installation.md` | Verification script | 10 min |
| 4 | `reproducibility.md` | Checklist + examples | 15 min |
| 5 | `oil_authentication.md` | Full worked example | 30 min |
| 6 | `data_format.md` | CSV spec + examples | 25 min |
| 7 | `reproducibility_checklist.md` | Narrative + tables | 20 min |
| 8 | `end_to_end_pipeline.md` | Real-world example | 20 min |
| 9 | `getting_started.md` | Hello world | 15 min |
| 10 | `faq_basic.md` | Code examples | 25 min |

---

## 📝 Page 1: `quickstart_15min.md` — Add Python Option

**Insert after "## Step 3: Run Your First Analysis":**

```python
### Option B: Python API

from foodspec.datasets import load_oil_example_data
from foodspec.preprocess import baseline_als, normalize_snv
from foodspec.ml import ClassifierFactory
from foodspec.validation import run_stratified_cv

spectra = load_oil_example_data()
spectra = baseline_als(spectra)
spectra = normalize_snv(spectra)
model = ClassifierFactory.create("random_forest", n_estimators=100)
metrics = run_stratified_cv(model, spectra.data, spectra.labels, cv=5)
print(f"✅ Accuracy: {metrics['accuracy']:.1%}")
```

**Expected Output:** `✅ Accuracy: 95.2%`

---

## 📝 Page 2: `first-steps_cli.md` — Add Context Block

**Insert at top (replace the first 20 lines):**

```markdown
# First Steps (CLI)

**Purpose:** Run complete analysis from terminal without Python code.
**Audience:** Researchers preferring shell commands and reproducible scripts.
**Time:** 5-10 minutes.
**Prerequisites:** FoodSpec installed; basic terminal knowledge.
```

---

## ✅ Validation Checklist

- [ ] Run `mkdocs build --strict` (zero warnings)
- [ ] Every code example has expected output
- [ ] All Next Steps links use relative paths (`../path/to/page.md`)
- [ ] No broken references (`grep -r "]\(" docs/ | grep -v "http"`)
- [ ] Context block on every page (Purpose, Audience, Time, Prerequisites)
- [ ] One H1 per page (check: `grep "^# " page.md | wc -l`)

---

## 🚀 Execute (Copy-Paste Commands)

```bash
# 1. Go to repo
cd /home/cs/FoodSpec

# 2. Edit first 2 pages (use VS Code find-replace or editor)
# See JOSS_DOCUMENTATION_UPGRADE_PLAN.md Section C for exact text

# 3. Verify build
mkdocs build --strict

# 4. Test imports
python -c "from foodspec.datasets import load_oil_example_data; print('✅')"

# 5. View docs locally
mkdocs serve  # Open http://127.0.0.1:8000
```

---

## 📞 Quick Help

**Q: Where's the template?**  
A: [JOSS_DOCUMENTATION_UPGRADE_PLAN.md](JOSS_DOCUMENTATION_UPGRADE_PLAN.md) Section A

**Q: What exact text do I paste?**  
A: [JOSS_DOCUMENTATION_UPGRADE_PLAN.md](JOSS_DOCUMENTATION_UPGRADE_PLAN.md) Section C (first 2 pages)

**Q: How do I test it worked?**  
A: Run `mkdocs build --strict` (must pass with zero warnings)

**Q: What if I have issues?**  
A: Check [reproducibility.md](docs/reproducibility.md) for debugging

---

## 🎓 Why This Matters for JOSS

| Criteria | What JOSS Reviewers Check |
|----------|---------------------------|
| **Usability** | Can I run the quickstart in 15 minutes? |
| **Documentation** | Is every page clear about WHO/WHY/WHEN/HOW? |
| **Reproducibility** | Are examples copy-pasteable and reproducible? |
| **Transparency** | Can I find what I need quickly? |

**Your upgrade directly addresses all 4.**

---

## 📅 Timeline

- **Day 1:** Apply Sections A-C (45 min for blockers)
- **Day 2:** Upgrade pages 3-10 (2 hours)
- **Day 3:** Final review + `mkdocs build --strict` pass
- **Ready:** JOSS submission by January 2026 ✅

---

**Document:** JOSS_DOCUMENTATION_UPGRADE_PLAN.md  
**Status:** Ready to execute  
**Last Updated:** 2025-12-30
