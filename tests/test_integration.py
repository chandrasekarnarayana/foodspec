"""Additional comprehensive tests targeting specific uncovered modules."""

import numpy as np
import pandas as pd

from foodspec.core.dataset import FoodSpectrumSet, from_sklearn, to_sklearn
from foodspec.features.fingerprint import correlation_similarity_matrix, cosine_similarity_matrix
from foodspec.features.peak_stats import compute_peak_stats, compute_ratio_table
from foodspec.features.ratios import compute_ratios
from foodspec.qc.novelty import novelty_score_single, novelty_scores
from foodspec.qc.prediction_qc import PredictionQCResult, evaluate_prediction_qc
from foodspec.synthetic.spectra import PeakSpec, generate_synthetic_ftir_spectrum, generate_synthetic_raman_spectrum


def test_to_sklearn_conversion():
    """Test converting FoodSpectrumSet to sklearn format."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(10, 100)
    meta = pd.DataFrame({"id": range(10), "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    ds = FoodSpectrumSet(x, wn, meta, modality="raman", label_col="label")

    X, y = to_sklearn(ds, label_col="label")

    assert X.shape == (10, 100)
    assert len(y) == 10


def test_from_sklearn_conversion():
    """Test creating FoodSpectrumSet from sklearn format."""
    X = np.random.randn(10, 100)
    y = np.array([0, 1] * 5)
    wn = np.linspace(500, 1500, 100)

    ds = from_sklearn(X, y, wn, modality="raman", labels_name="label")

    assert ds.x.shape == (10, 100)
    assert "label" in ds.metadata.columns
    assert len(ds.metadata) == 10


def test_cosine_similarity_matrix():
    """Test computing cosine similarity matrix."""
    X_ref = np.random.randn(5, 100)
    X_query = np.random.randn(3, 100)

    sim = cosine_similarity_matrix(X_ref, X_query)

    assert sim.shape == (5, 3)
    # Similarities should be in [-1, 1]
    assert np.all(sim >= -1) and np.all(sim <= 1)


def test_correlation_similarity_matrix():
    """Test computing correlation similarity matrix."""
    X_ref = np.random.randn(5, 100)
    X_query = np.random.randn(3, 100)

    corr = correlation_similarity_matrix(X_ref, X_query)

    assert corr.shape == (5, 3)
    # Correlations should be in [-1, 1]
    assert np.all(corr >= -1.1) and np.all(corr <= 1.1)  # Allow small numerical errors


def test_compute_peak_stats():
    """Test computing peak statistics."""
    peak_data = pd.DataFrame(
        {
            "spectrum_id": [1, 1, 2, 2, 3],
            "peak_id": ["p1", "p2", "p1", "p2", "p1"],
            "position": [1000.0, 1200.0, 1005.0, 1198.0, 1002.0],
            "intensity": [10.0, 8.0, 12.0, 9.0, 11.0],
        }
    )

    stats = compute_peak_stats(peak_data, metadata=None)

    assert isinstance(stats, pd.DataFrame)
    assert len(stats) > 0
    assert "mean_pos" in stats.columns


def test_compute_ratio_table():
    """Test computing ratio table."""
    ratio_data = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "R_1000_1200": [1.25, 1.33, 1.29],
            "R_1400_1600": [0.95, 1.05, 1.00],
        }
    )

    result = compute_ratio_table(ratio_data, metadata=None)

    assert isinstance(result, pd.DataFrame)


def test_compute_ratios():
    """Test computing ratios from dataframe."""
    df = pd.DataFrame(
        {
            "sample_id": [1, 2, 3],
            "I_1000": [10.0, 12.0, 11.0],
            "I_1200": [5.0, 6.0, 5.5],
            "I_1400": [8.0, 9.0, 8.5],
        }
    )

    ratio_def = {
        "R_1000_1200": ("I_1000", "I_1200"),
        "R_1400_1200": ("I_1400", "I_1200"),
    }

    result = compute_ratios(df, ratio_def)

    assert "R_1000_1200" in result.columns
    assert "R_1400_1200" in result.columns
    assert len(result) == 3


def test_evaluate_prediction_qc():
    """Test evaluating prediction QC."""
    # Probabilities for a 2-class prediction
    probs = np.array([0.9, 0.8, 0.7])

    qc_result = evaluate_prediction_qc(probs, drift_score=0.1, ece=0.05)

    assert isinstance(qc_result, PredictionQCResult)
    assert hasattr(qc_result, "do_not_trust")


def test_novelty_scores():
    """Test computing novelty scores."""
    X_train = np.random.randn(20, 50)
    X_test = np.random.randn(5, 50)
    X_test[0] += 10  # Make one sample very different

    scores, flags = novelty_scores(X_train, X_test, metric="euclidean")

    assert len(scores) == 5
    assert len(flags) == 5
    assert scores[0] > scores[1]  # Outlier should have higher score


def test_novelty_score_single():
    """Test computing novelty score for single sample."""
    X_train = np.random.randn(20, 50)
    x_test = np.random.randn(50)

    score, is_novel = novelty_score_single(X_train, x_test, metric="euclidean")

    assert isinstance(score, (float, np.floating))
    assert isinstance(is_novel, (bool, np.bool_))
    assert score >= 0


def test_generate_synthetic_raman_spectrum():
    """Test generating synthetic Raman spectrum."""
    peaks = [PeakSpec(position=800, amplitude=10.0, width=20), PeakSpec(position=1200, amplitude=8.0, width=15)]

    wn, spectrum = generate_synthetic_raman_spectrum(peaks=peaks, noise_level=0.1)

    assert len(spectrum) == len(wn)
    assert np.any(spectrum > 0)


def test_generate_synthetic_ftir_spectrum():
    """Test generating synthetic FTIR spectrum."""
    bands = [PeakSpec(position=1000, amplitude=0.5, width=20), PeakSpec(position=1600, amplitude=0.7, width=30)]

    wn, spectrum = generate_synthetic_ftir_spectrum(bands=bands, noise_level=0.05)

    assert len(spectrum) == len(wn)
    # FTIR is absorbance, so should have positive values
    assert np.any(spectrum > 0)


def test_foodspectrumset_label_col_usage():
    """Test FoodSpectrumSet with label_col."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(10, 100)
    meta = pd.DataFrame({"id": range(10), "class": ["A"] * 5 + ["B"] * 5})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman", label_col="class")

    assert ds.label_col == "class"
    assert "class" in ds.metadata.columns


def test_foodspectrumset_group_col_usage():
    """Test FoodSpectrumSet with group_col."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(10, 100)
    meta = pd.DataFrame({"id": range(10), "group": ["G1"] * 5 + ["G2"] * 5})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman", group_col="group")

    assert ds.group_col == "group"


def test_foodspectrumset_batch_col_usage():
    """Test FoodSpectrumSet with batch_col."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(10, 100)
    meta = pd.DataFrame({"id": range(10), "batch": ["B1"] * 5 + ["B2"] * 5})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman", batch_col="batch")

    assert ds.batch_col == "batch"


def test_foodspectrumset_edge_case_small():
    """Test FoodSpectrumSet with minimum valid size."""
    wn = np.array([500.0, 600.0, 700.0])  # Exactly 3 points (minimum)
    x = np.random.randn(2, 3)
    meta = pd.DataFrame({"id": [1, 2]})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")

    assert ds.x.shape == (2, 3)


def test_foodspectrumset_repr():
    """Test FoodSpectrumSet string representation."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5)})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")
    repr_str = repr(ds)

    assert "FoodSpectrumSet" in repr_str
    assert "5" in repr_str  # Number of samples


def test_foodspectrumset_len():
    """Test FoodSpectrumSet length."""
    wn = np.linspace(500, 1500, 100)
    x = np.random.randn(5, 100)
    meta = pd.DataFrame({"id": range(5)})

    ds = FoodSpectrumSet(x, wn, meta, modality="raman")

    assert len(ds) == 5
