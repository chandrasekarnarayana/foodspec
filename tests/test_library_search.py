import numpy as np
from foodspec.library_search import search_library, overlay_plot


def test_search_library_topk_confidence():
    # Build a tiny library with a close match
    wn = np.linspace(1000, 1020, 5)
    query = np.array([1, 2, 3, 4, 5]).astype(float)
    lib = np.stack([
        query + 0.01,  # closest
        query[::-1],   # far
        query + 1.0,   # medium
    ])
    labels = ["close", "reverse", "shifted"]
    matches = search_library(query, lib, labels=labels, k=2, metric="cosine")
    assert len(matches) == 2
    assert matches[0].label == "close"
    assert 0 <= matches[0].confidence <= 1

    # Overlay plot should return a figure
    fig = overlay_plot(query, wn, [(m.label, lib[m.index]) for m in matches])
    assert fig is not None
