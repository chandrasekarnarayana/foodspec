"""
Tests for visualization modules to improve coverage.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from foodspec.core.dataset import FoodSpectrumSet


@pytest.fixture
def sample_spectrum_set():
    """Create sample FoodSpectrumSet for testing."""
    wn = np.linspace(500, 1500, 100)
    spectra = np.random.randn(10, 100) + 10
    metadata = pd.DataFrame(
        {
            "sample_id": range(10),
            "group": ["A"] * 5 + ["B"] * 5,
            "batch": ["batch1"] * 3 + ["batch2"] * 3 + ["batch3"] * 4,
        }
    )
    return FoodSpectrumSet(spectra, wavenumbers=wn, metadata=metadata)


def test_plot_spectra_basic(sample_spectrum_set):
    """Test basic spectra plotting."""
    from foodspec.viz.spectra import plot_spectra

    fig, ax = plt.subplots()
    plot_spectra(sample_spectrum_set, ax=ax)
    assert ax is not None
    plt.close(fig)


def test_plot_spectra_with_grouping(sample_spectrum_set):
    """Test spectra plotting with color grouping."""
    from foodspec.viz.spectra import plot_spectra

    fig, ax = plt.subplots()
    plot_spectra(sample_spectrum_set, color_by="group", ax=ax)
    assert ax is not None
    plt.close(fig)


def test_plot_mean_spectrum(sample_spectrum_set):
    """Test mean spectrum plotting."""
    from foodspec.viz.spectra import plot_mean_spectrum

    fig, ax = plt.subplots()
    plot_mean_spectrum(sample_spectrum_set, ax=ax)
    assert ax is not None
    plt.close(fig)


def test_plot_confusion_matrix_basic():
    """Test confusion matrix plotting."""
    from foodspec.viz.classification import plot_confusion_matrix

    cm = np.array([[45, 5], [3, 47]])
    labels = ["Class A", "Class B"]

    fig, ax = plt.subplots()
    plot_confusion_matrix(cm, labels, ax=ax)
    assert ax is not None
    plt.close(fig)
