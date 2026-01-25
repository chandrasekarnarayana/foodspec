"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.
"""

from pathlib import Path

import numpy as np

from foodspec.core.artifacts import ArtifactRegistry
from foodspec.viz import (
    plot_confusion_matrix,
    plot_heatmap,
    plot_pca_scatter,
    plot_raw_vs_processed,
    plot_reliability_diagram,
)


def _artifacts(tmp_path: Path) -> ArtifactRegistry:
    reg = ArtifactRegistry(tmp_path)
    reg.ensure_layout()
    return reg


def test_plot_raw_vs_processed_creates_file(tmp_path: Path) -> None:
    wn = np.linspace(400, 1800, 10)
    raw = np.sin(wn / 100.0)
    proc = raw * 0.5
    artifacts = _artifacts(tmp_path)

    path = plot_raw_vs_processed(wn, raw, proc, artifacts)
    assert path.exists()


def test_plot_heatmap_creates_file(tmp_path: Path) -> None:
    mat = np.arange(9).reshape(3, 3)
    artifacts = _artifacts(tmp_path)

    path = plot_heatmap(mat, artifacts, xlabels=["a", "b", "c"], ylabels=["x", "y", "z"])
    assert path.exists()


def test_plot_pca_scatter_creates_file(tmp_path: Path) -> None:
    scores = np.array([[0.1, 0.2], [0.5, 0.4], [0.2, 0.8]])
    labels = np.array([0, 1, 0])
    artifacts = _artifacts(tmp_path)

    path = plot_pca_scatter(scores, labels, artifacts)
    assert path.exists()


def test_plot_confusion_matrix_creates_file(tmp_path: Path) -> None:
    cm = np.array([[8, 2], [1, 9]])
    artifacts = _artifacts(tmp_path)

    path = plot_confusion_matrix(cm, class_names=["neg", "pos"], artifacts=artifacts)
    assert path.exists()


def test_plot_reliability_diagram_creates_file(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 0, 1, 0])
    proba = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4],
        [0.4, 0.6],
        [0.9, 0.1],
    ])
    artifacts = _artifacts(tmp_path)

    path = plot_reliability_diagram(y_true, proba, artifacts, n_bins=5)
    assert path.exists()
