"""
Quickstart script for oil authentication using foodspec.
Run with: python examples/oil_authentication_quickstart.py
"""

import matplotlib.pyplot as plt

from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.chemometrics.pca import run_pca
from foodspec.data.loader import load_example_oils
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.viz.pca import plot_pca_scores


def main():
    fs = load_example_oils()
    result = run_oil_authentication_quickstart(fs, label_column="oil_type")

    # Print metrics summary
    print(result.cv_metrics)

    # Plot confusion matrix
    ax_cm = plot_confusion_matrix(result.confusion_matrix, result.class_labels)
    fig_cm = ax_cm.get_figure()
    fig_cm.savefig("outputs/oil_auth_confusion.png", dpi=150, bbox_inches="tight")

    # PCA visualization on preprocessed spectra
    pca, res = run_pca(fs.x, n_components=2)
    ax_pca = plot_pca_scores(res.scores, labels=fs.metadata["oil_type"])
    fig_pca = ax_pca.get_figure()
    fig_pca.savefig("outputs/oil_auth_pca.png", dpi=150, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
