# Examples Gallery

Welcome to the FoodSpec Examples Gallery! This section contains canonical, runnable examples demonstrating FoodSpec's core capabilities. Each example includes:

- **Learning objectives**: What you'll understand after completing the example
- **Prerequisites**: Required knowledge and dependencies
- **Copy-paste code blocks**: Small, digestible code snippets you can run immediately
- **Links to full scripts**: Complete runnable code in `/examples/`
- **Generated figures**: Publication-quality visualizations from real FoodSpec workflows

---

## Quick Navigation

### Beginner Examples (Start here!)

**[Oil Authentication](01_oil_authentication.md)** – Supervised Classification  
Learn how to build a classifier to distinguish olive oils from adulterants using Raman spectroscopy. Topics: cross-validation, confusion matrices, model discrimination.

**[Heating Quality Monitoring](02_heating_quality_monitoring.md)** – Time-Series Analysis  
Analyze how cooking oils degrade over time. Extract chemical indicators, fit degradation models, and estimate shelf-life.

### Intermediate Examples (Build deeper understanding)

**[Mixture Analysis](03_mixture_analysis.md)** – Quantification via NNLS  
Learn spectral unmixing to quantify ingredient blends and detect adulterants using Non-Negative Least Squares.

**[Hyperspectral Mapping](04_hyperspectral_mapping.md)** – Spatial Analysis  
Work with 3D hyperspectral data. Segment images, extract regions of interest, and perform spatial quality assessments.

### Capstone Example (Mastery)

**[End-to-End Protocol Run](05_end_to_end_protocol_run.md)** – Unified FoodSpec API  
Master the complete FoodSpec workflow: quality check → preprocess → train → evaluate → export. Demonstrates best practices for reproducible, auditable science.

---

## Learning Path

| Level | Examples | Duration | Focus |
|-------|----------|----------|-------|
| **Beginner** | Oil Auth + Heating | 15 min | Classification, time-series |
| **Intermediate** | + Mixture + HSI | 45 min | Quantification, spatial data |
| **Capstone** | + End-to-End | 60 min | Complete workflows, reproducibility |

---

## Running Examples

### Option 1: Run Python Scripts Directly

All examples come with production-ready Python scripts in the `/examples/` directory:

```bash
cd foodspec
python examples/oil_authentication_quickstart.py
python examples/heating_quality_quickstart.py
# ... and more
```

### Option 2: Interactive Jupyter Notebooks

Each example has a teaching notebook with narrative, visualizations, and step-by-step explanations:

```bash
jupyter notebook examples/tutorials/01_oil_authentication_teaching.ipynb
jupyter notebook examples/tutorials/02_heating_stability_teaching.ipynb
# ... and more
```

### Option 3: View Minimal Code on This Site

Scroll down to any example page to see minimal, focused code blocks that demonstrate the key concepts.

---

## What's Included

### Example Scripts
- **Location**: `/examples/*.py`
- **Status**: Production-ready, tested, enhanced with docstrings
- **Runtime**: 1–15 seconds per example
- **Data**: All synthetic datasets included (no downloads needed)

### Teaching Notebooks
- **Location**: `/examples/tutorials/*.ipynb`
- **Format**: Jupyter notebooks with markdown narrative + executable code
- **Structure**: Learning objectives → background → walkthrough → visualization → key takeaways
- **Interactive**: Run cells, modify parameters, explore results

### Generated Figures
- **Confusion matrices**: Classification performance
- **Trend curves**: Degradation kinetics
- **Segmentation maps**: Spatial analysis results
- **PCA plots**: Data structure exploration

---

## Prerequisites

### Required Knowledge
- Basic Python (variables, functions, imports)
- NumPy/Pandas familiarity (arrays, DataFrames)
- Spectroscopy fundamentals (optional, we explain concepts)

### Required Software
- Python 3.10+
- FoodSpec (installed via `pip install foodspec`)
- Jupyter (for notebook-based learning)
- matplotlib (for visualizations, included with FoodSpec)

### Optional
- scikit-learn (for advanced model evaluation)
- Raman spectroscopy domain knowledge (we teach this!)

---

## Datasets & Data Files

All examples use **synthetic, deterministic datasets** included in the repository:

- `examples/data/oil_synthetic.csv` – Raman spectra of 4 oil types
- `examples/data/hsi_synthetic.npz` – 3D hyperspectral cube
- Automatically generated mixtures and time-series data in memory

**No data downloads needed.** All examples run offline with included files.

---

## Figure Gallery

### Oil Authentication
![Classification confusion matrix](https://github.com/chandrasekarnarayana/foodspec/raw/mahttps://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/oil_auth_confusion.png)

### Heating Quality
![Degradation trend curve](https://github.com/chandrasekarnarayana/foodspec/raw/mahttps://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/heating_ratio_vs_time.png)

### Hyperspectral Mapping
![Spatial segmentation map](https://github.com/chandrasekarnarayana/foodspec/raw/mahttps://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/hyperspectral_demo/hsi_label_map.png)

---

## Quick Stats

- **5 canonical examples**: Oil, heating, mixture, HSI, unified API
- **1,200+ lines of teaching notebooks**: Comprehensive walkthroughs
- **5 production scripts**: Tested, fixed, enhanced with documentation
- **22 seconds total runtime**: All examples complete in <30s
- **0 external data dependencies**: All datasets included

---

## Support & Contribution

**Questions or issues?**
- Check the [FAQ](../help/faq.md) and [Troubleshooting Guide](../help/troubleshooting.md)
- Search the [Glossary](../reference/glossary.md)

**Want to contribute an example?**
- See the [Contributing Guide](../developer-guide/contributing.md)
- Follow the [Documentation Guidelines](../developer-guide/documentation_guidelines.md)

**Learn more:**
- [Methods reference](../methods/index.md) – Technical deep-dives
- [Theory section](../theory/index.md) – Science foundations
- [Complete API](../api/index.md) – Full function reference

---

## Next Steps

1. **Start here**: Pick an example that matches your skill level
2. **Run it**: Copy code blocks or execute the full script
3. **Explore**: Modify parameters and observe changes
4. **Learn**: Read the surrounding explanation and generated figures
5. **Deepen**: Follow links to methods, workflows, and theory sections

Happy learning! 🚀

