import numpy as np
import pandas as pd

from foodspec.utils.troubleshooting import (
    estimate_snr,
    summarize_class_balance,
    detect_outliers,
    check_missing_metadata,
)


def test_estimate_snr_basic():
    spectrum = [0, 1, 2, 3, 4, 5]
    snr = estimate_snr(spectrum)
    assert snr > 0  # heuristic; just ensure finite positive


def test_summarize_class_balance():
    labels = ["a", "b", "a", "c"]
    counts = summarize_class_balance(labels)
    assert counts["a"] == 2
    assert set(counts.index) == {"a", "b", "c"}


def test_detect_outliers():
    X = np.vstack(
        [
            np.ones((10, 3)),  # cluster
            np.array([[10, 10, 10]]),  # outlier
        ]
    )
    mask = detect_outliers(X, n_components=2, z_thresh=2.5)
    assert mask.sum() == 1


def test_check_missing_metadata():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    missing = check_missing_metadata(df, ["a", "c"])
    assert missing == ["c"]
