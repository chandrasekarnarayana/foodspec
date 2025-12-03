"""
Quickstart script for oil authentication using foodspec.
Run with: python examples/oil_authentication_quickstart.py
"""

from foodspec.data.loader import load_example_oils
from foodspec.apps.oils import run_oil_authentication_quickstart
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.chemometrics.pca import run_pca
from foodspec.viz.pca import plot_pca_scores
import matplotlib.pyplot as plt


def main():
    fs = load_example_oils()
    result = run_oil_authentication_quickstart(fs, label_column="oil_type")

    # Print metrics summary
    print(result.cv_metrics)

    # Plot confusion matrix
    fig_cm = plot_confusion_matrix(result.confusion_matrix, result.class_labels)
    fig_cm.savefig("oil_auth_confusion.png", dpi=150)

    # PCA visualization on preprocessed spectra
    pca, res = run_pca(fs.x, n_components=2)
    fig_scores = plot_pca_scores(res.scores, labels=fs.metadata["oil_type"])
    fig_scores.savefig("oil_auth_pca.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    main()
