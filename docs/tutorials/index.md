# Tutorials: Beginner → Advanced Learning Path

Step-by-step tutorials from first plots to reproducible pipelines, with proper validation and clear success criteria.

---

## 🎯 Learning Paths

### Beginner (5–15 min each)
Get your first FoodSpec analysis running. No prior knowledge required.

| Tutorial | Time | What You'll Learn |
|----------|------|------------------|
| [Load Spectra & Plot](beginner/01-load-and-plot.md) | 5 min | Load CSV data, create basic plots, understand spectral format |
| [Baseline Correction & Smoothing](beginner/02-preprocess.md) | 10 min | Clean noisy spectra using ALS baseline and Savitzky–Golay smoothing |
| [Simple Classification](beginner/03-classify.md) | 15 min | Classify oil types using PCA + logistic regression; visualize results |

---

### Intermediate (20–40 min each)
Solve real-world problems with proper validation and domain knowledge.

| Tutorial | Time | What You'll Learn |
|----------|------|------------------|
| [Oil Authentication with Validation](intermediate/01-oil-authentication.md) | 25 min | Cross-validation, confusion matrices, reproducible protocols |
| [Domain Shift: Oil vs Chips](intermediate/02-matrix-effects.md) | 35 min | Matrix effects, divergence analysis, marker stability across matrices |
| [Stability Tracking](intermediate/03-stability.md) | 30 min | Monitor degradation/aging using time series and QC metrics |

---

### Advanced (45–90 min each)
Build reproducible, production-ready pipelines with experiment tracking.

| Tutorial | Time | What You'll Learn |
|----------|------|------------------|
| [Reproducible Pipelines with Configs](advanced/01-reproducible-pipelines.md) | 45 min | YAML protocol design, version control, experiment tracking |
| [Reference Workflow: Oil Authentication](advanced/02-reference-workflow.md) | 90 min | Canonical reproducible workflow; template for publications |
| [HSI Surface Mapping](advanced/03-hsi-mapping.md) | 60 min | Hyperspectral mapping and visualization | 

---

## 📚 Supplementary Resources

- [Examples Gallery](../examples_gallery.md) — Practical, runnable examples
- [End-to-End Notebooks](end-to-end-notebooks.md) — Interactive Jupyter notebooks
- [Protocols & YAML](../user-guide/protocols_and_yaml.md) — Deep dive into protocol configuration
- [Troubleshooting Guide](../troubleshooting/common_problems_and_solutions.md) — Fix common errors

---

## 🎓 Tutorial Template

Every tutorial in FoodSpec follows a consistent structure:

1. **Goal** — What problem are we solving?
2. **Data** — What data format and size do we need?
3. **Steps** — High-level workflow overview
4. **Code** — Copy-paste runnable Python/CLI examples
5. **Results** — Expected outputs, plots, and metrics
6. **Interpretation** — How to read the results
7. **Pitfalls** — Common mistakes and how to avoid them
8. **Next Steps** — What to learn next

---

## 🚀 Quick Navigation

### By Use Case
- Authenticate oils → [Oil Authentication](intermediate/01-oil-authentication.md)
- Handle matrix effects → [Oil vs Chips](intermediate/02-matrix-effects.md)
- Build a production pipeline → [Reproducible Pipelines](advanced/01-reproducible-pipelines.md)
- Map surfaces with HSI → [HSI Surface Mapping](advanced/03-hsi-mapping.md)

### By Skill Level
- New to FoodSpec → Start with Beginner
- Comfortable with basics → Try Intermediate
- Publishing results → Dive into Advanced

---

## ✅ Prerequisites

### Level 1
- Python 3.10+ installed
- FoodSpec installed: `pip install foodspec`
- ~10 minutes of your time

### Level 2
- Complete Level 1 tutorials (or FoodSpec basics)
- Understanding of cross-validation and classification metrics
- Your own data (or use synthetic examples)

### Level 3
- Complete Level 2 tutorials (or publication experience)
- Familiarity with YAML configuration
- Git for version control (recommended)

---

## 🔗 Connection to Other Docs

- [Getting Started](../getting-started/index.md) — Installation and quickstarts
- [Foundations](../foundations/index.md) — Data structures and key concepts
- [Workflows](../workflows/index.md) — Domain-specific analysis patterns
- [Theory & Background](../theory/spectroscopy_basics.md) — Scientific principles

---

## 💡 Tips for Success

1. **Type the code yourself** (don't copy-paste) to build muscle memory
2. **Experiment with parameters** — Change smoothing window size, regularization, etc.
3. **Check the troubleshooting section** before asking for help
4. **Link to tutorials when sharing work** — Shows your methodology is reproducible
5. **Join the community** — [GitHub Discussions](https://github.com/chandrasekarnarayana/foodspec/discussions)

---

## 📊 Progressive Difficulty

```plaintext
Level 1 (Beginner)       Level 2 (Applied)          Level 3 (Advanced)
├─ Load spectra          ├─ Cross-validation        ├─ Experiment tracking
├─ Plot basics           ├─ Domain shift             ├─ Reproducible configs
└─ Simple classifier     └─ Model comparison        └─ Publication pipeline
     ↓                          ↓                            ↓
  5-15 min             20-40 min                  45-90 min
  No assumptions        FoodSpec basics            Production-ready
```

---

## 🎯 Success Criteria

After each level, you should be able to:

**Level 1 Complete?** ✓
- [ ] Load your own CSV spectra
- [ ] Preprocess and visualize them
- [ ] Train and evaluate a simple classifier

**Level 2 Complete?** ✓
- [ ] Validate models using cross-validation
- [ ] Interpret metrics and confusion matrices
- [ ] Identify domain shift issues

**Level 3 Complete?** ✓
- [ ] Define protocols in YAML
- [ ] Track experiments and versions
- [ ] Generate publication-ready reports

---

## 🐛 Got Stuck?

- Check the **Pitfalls** section in each tutorial
- See [Troubleshooting FAQ](../troubleshooting/troubleshooting_faq.md)
- [Report an issue](https://github.com/chandrasekarnarayana/foodspec/issues)

Happy learning! 🎓
