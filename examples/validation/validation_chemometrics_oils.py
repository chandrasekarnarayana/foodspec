# %% [markdown]
# # Validation: Chemometrics on Synthetic Oils
#
# Demonstrates foodspec oil authentication workflow and PCA visualization.

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from foodspec.apps.oils import run_oil_authentication_workflow
from foodspec.chemometrics.pca import run_pca
from foodspec.data.loader import load_example_oils
from foodspec.viz.classification import plot_confusion_matrix
from foodspec.viz.pca import plot_pca_scores


def main():
    spectra = load_example_oils()

    # Oil authentication workflow
    result = run_oil_authentication_workflow(spectra, label_column="oil_type", classifier_name="rf", cv_splits=5)
    print("Cross-validation metrics:")
    print(result.cv_metrics)
    print("Confusion matrix:\n", result.confusion_matrix)

    fig, ax = plt.subplots(figsize=(4, 4))
    plot_confusion_matrix(result.confusion_matrix, class_names=result.class_labels, ax=ax)
    fig.tight_layout()
    plt.savefig("validation_oils_confusion.png", dpi=150)
    plt.show()

    # PCA visualization
    pca_model, pca_res = run_pca(spectra.x, n_components=2)
    labels = spectra.metadata["oil_type"].to_numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_pca_scores(pca_res.scores, labels=labels, ax=ax)
    ax.set_title("PCA scores (oil_type)")
    fig.tight_layout()
    plt.savefig("validation_oils_pca.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()

