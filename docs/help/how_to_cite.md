# How to Cite FoodSpec

**Purpose:** Get citation formats for FoodSpec and datasets used with it.  
**Audience:** Researchers publishing results using FoodSpec.  
**Time to read:** 2–3 minutes.  
**Prerequisites:** None.

---

## Recommended Citation

If you use FoodSpec in your research, please cite it as:

```bibtex
@software{narayana2024foodspec,
  title = {foodspec: A Python toolkit for Raman and FTIR spectroscopy in food science},
  author = {Narayana, Chandrasekar Subramani},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/chandrasekarnarayana/foodspec},
  doi = {TBD},
  note = {Retrieved from \url{https://pypi.org/project/foodspec/}}
}
```

### APA Format

Narayana, C. S. (2024). *foodspec: A Python toolkit for Raman and FTIR spectroscopy in food science* (Version 1.0.0). GitHub. https://github.com/chandrasekarnarayana/foodspec

### MLA Format

Narayana, Chandrasekar Subramani. *foodspec: A Python toolkit for Raman and FTIR spectroscopy in food science*. Version 1.0.0, GitHub, 2024, github.com/chandrasekarnarayana/foodspec.

### Plain Text

Narayana, C. S. (2024). foodspec: A Python toolkit for Raman and FTIR spectroscopy in food science (v1.0.0). https://github.com/chandrasekarnarayana/foodspec

---

## Version-Specific Citations

### Citing a Specific Release

Replace the version number in the formats above. For example, cite v0.2.0:

```bibtex
@software{narayana2024foodspec,
  title = {foodspec},
  version = {0.2.0},
  url = {https://github.com/chandrasekarnarayana/foodspec/releases/tag/v0.2.0}
}
```

**GitHub releases:** https://github.com/chandrasekarnarayana/foodspec/releases

**PyPI versions:** https://pypi.org/project/foodspec/#history

### DOI (When Available)

Once FoodSpec is published or archived, a DOI will be available. You can:
- Register a version with [Zenodo](https://zenodo.org/) for a citable DOI
- Include the DOI in citations: `doi: 10.5281/zenodo.XXXXXXX` (example format)

---

## Citing Datasets Used with FoodSpec

If you use FoodSpec to analyze published datasets, cite **both** FoodSpec and the dataset:

### In Methods Section

> We acquired Raman spectra and analyzed them using FoodSpec v1.0.0 [Narayana, 2024]. Data were processed with baseline correction (ALS), smoothing (Savitzky–Golay), and normalization (L2). Raw spectra are available at [Dataset DOI].

### Data Citation Template

```bibtex
@dataset{original_authors_year,
  title = {[Dataset Name]},
  author = {[Authors]},
  year = {[Year]},
  doi = {[Dataset DOI]},
  url = {[Repository URL]},
  note = {Version [Version number]}
}
```

### Finding Dataset DOIs

- **Zenodo:** https://zenodo.org/
- **GitHub:** Use [Zenodo GitHub integration](https://guides.github.com/features/pages/) to create DOIs
- **figshare:** https://figshare.com/
- **OSF (Open Science Framework):** https://osf.io/
- **Kaggle:** https://www.kaggle.com/datasets/

---

## Software and Dependencies

If you use specific FoodSpec workflows or depend heavily on underlying libraries, you may also cite:

- **scikit-learn** (classifiers, PCA): Pedregosa et al., 2011
- **NumPy/SciPy** (numerical computing): Harris et al., 2020; Virtanen et al., 2020
- **matplotlib** (visualization): Hunter, 2007

See [metrics reference](../reference/metrics_reference.md) and [method comparison](../reference/method_comparison.md) for full citations.

---

## Citing in Code and Documentation

In your analysis scripts or supplementary code, include a comment:

```python
# This analysis uses FoodSpec v1.0.0
# Citation: Narayana, C. S. (2024). foodspec: A Python toolkit for 
#           Raman and FTIR spectroscopy in food science. v1.0.0.
#           https://github.com/chandrasekarnarayana/foodspec

from foodspec.io import load_csv
from foodspec.preprocess import baseline_als
# ... rest of analysis
```

---

## Next Steps

- [Reproducibility Checklist](../protocols/reproducibility_checklist.md) — Document your full analysis
- [Reporting Guidelines](../troubleshooting/reporting_guidelines.md) — Write methods section
- [FAIR Principles](../theory/data_structures_and_fair_principles.md) — Share data and code openly
