"""Generate PCA, loadings, and t-SNE figures with embedding metrics.

This script produces three figures for the documentation:
1) PCA scores plot coloured by class
2) PCA loadings plot (PC1/PC2) on the wavenumber axis
3) t-SNE 2D embedding coloured by class

It also prints the silhouette score and a simple between/within scatter ratio
computed on the PCA scores, using the helper functions in ``foodspec.metrics``.

Output files are written to ``docs/assets``:
    - pca_scores.png
    - pca_loadings.png
    - tsne_scores.png

The synthetic spectral dataset is generated with foodspec.synthetic to avoid
external data dependencies. Two classes are created by shifting the position
and intensity of one peak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from foodspec.metrics import (
    compute_between_within_ratio,
    compute_embedding_silhouette,
)
from foodspec.synthetic import generate_synthetic_raman_spectrum
from foodspec.synthetic.spectra import PeakSpec

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


def build_synthetic_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a small synthetic two-class spectral dataset.

    Returns
    -------
    wavenumbers : np.ndarray
        Shared wavenumber axis (cm^-1).
    spectra : np.ndarray
        Spectra matrix of shape (n_samples, n_wavenumbers).
    labels : np.ndarray
        Integer labels (0/1) for the two classes.
    """

    rng = np.random.default_rng(42)
    n_samples_per_class = 25

    # Base peaks shared between classes
    base_peaks = [
        PeakSpec(1000.0, 0.8, 25.0),
        PeakSpec(1450.0, 0.6, 35.0),
        PeakSpec(1650.0, 0.75, 30.0),
    ]

    spectra = []
    labels = []
    wavenumbers = None

    # Class 0: base peaks
    for _ in range(n_samples_per_class):
        np.random.seed(int(rng.integers(0, 10_000)))
        wn, spec = generate_synthetic_raman_spectrum(
            peaks=base_peaks,
            noise_level=1.5,
        )
        wavenumbers = wn
        spectra.append(spec)
        labels.append(0)

    # Class 1: shift one peak slightly and increase intensity
    shifted_peaks = [
        PeakSpec(1000.0, 0.8, 25.0),
        PeakSpec(1458.0, 0.9, 35.0),
        PeakSpec(1650.0, 0.95, 30.0),
    ]
    for _ in range(n_samples_per_class):
        np.random.seed(int(rng.integers(0, 10_000)))
        wn, spec = generate_synthetic_raman_spectrum(
            peaks=shifted_peaks,
            noise_level=1.5,
        )
        spectra.append(spec)
        labels.append(1)

    return wavenumbers, np.asarray(spectra), np.asarray(labels)


def make_pca_figures(wavenumbers: np.ndarray, spectra: np.ndarray, labels: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(spectra)

    # Metrics on PCA space
    sil = compute_embedding_silhouette(scores, labels)
    bw = compute_between_within_ratio(scores, labels)
    print(f"Silhouette (PCA scores): {sil:.3f}")
    print(f"Between/Within ratio (PCA scores): {bw:.3f}")

    # Scores plot
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap="tab10", alpha=0.8)
    ax.set_xlabel("PC1 scores")
    ax.set_ylabel("PC2 scores")
    ax.set_title("PCA scores (synthetic Raman dataset)")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, ["Class 0", "Class 1"], title="Class")
    ax.text(0.02, 0.02, f"sil={sil:.2f}\nb/w={bw:.2f}", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "pca_scores.png", dpi=200)
    plt.close(fig)

    # Loadings plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wavenumbers, pca.components_[0], label="PC1 loadings")
    ax.plot(wavenumbers, pca.components_[1], label="PC2 loadings", alpha=0.8)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Loading weight")
    ax.set_title("PCA loadings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "pca_loadings.png", dpi=200)
    plt.close(fig)


def make_tsne_figure(scores: np.ndarray, labels: np.ndarray) -> None:
    tsne = TSNE(n_components=2, perplexity=20, learning_rate="auto", random_state=42, init="pca")
    emb = tsne.fit_transform(scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", alpha=0.85)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE embedding (from PCA scores)")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, ["Class 0", "Class 1"], title="Class")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "tsne_scores.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    wavenumbers, spectra, labels = build_synthetic_dataset()
    make_pca_figures(wavenumbers, spectra, labels)

    # Reuse PCA scores for t-SNE initialisation (more stable) to keep runtime low
    pca = PCA(n_components=10, random_state=42)
    scores = pca.fit_transform(spectra)
    make_tsne_figure(scores, labels)


if __name__ == "__main__":
    main()
