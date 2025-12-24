# Spectral Library Search

Find the closest spectra in a library to answer “what does this spectrum resemble?”. Supports similarity metrics: cosine, Pearson correlation, Euclidean (negative distance), SID (negative divergence), and SAM (negative angle).

## CLI

```bash
foodspec-library-search --query query.csv --library lib.csv --label-col label --k 5 --metric cosine --overlay-out overlay.png
```

- `query.csv`: one row containing intensities across numeric wavenumber columns (e.g., `1000,1005,1010,...`).
- `lib.csv`: multiple rows with the same wavenumber columns and optional `label` column.
- `--metric`: `cosine|pearson|euclidean|sid|sam`.
- `--overlay-out`: saves an overlay plot of the query vs top-k matches.

## Python API

```python
import numpy as np
from foodspec.library_search import search_library, overlay_plot

# Wavenumber axis and data
wn = np.linspace(1000, 1020, 5)
query = np.array([1, 2, 3, 4, 5], dtype=float)
library = np.stack([query + 0.01, query[::-1], query + 1.0])
labels = ["close", "reverse", "shifted"]

matches = search_library(query, library, labels=labels, k=2, metric="cosine")
fig = overlay_plot(query, wn, [(m.label, library[m.index]) for m in matches])
fig.savefig("overlay.png")
```
