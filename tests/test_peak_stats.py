import pandas as pd

from foodspec.features.peak_stats import compute_peak_stats, compute_ratio_table


def test_compute_peak_stats_groups():
    peaks = pd.DataFrame(
        {
            "spectrum_id": ["s1", "s2", "s3", "s4"],
            "peak_id": ["p1", "p1", "p1", "p1"],
            "position": [1654, 1656, 1655, 1657],
            "intensity": [10.0, 12.0, 11.0, 13.0],
        }
    )
    meta = pd.DataFrame({"spectrum_id": ["s1", "s2", "s3", "s4"], "group": ["A", "A", "B", "B"]})
    stats = compute_peak_stats(peaks, metadata=meta, group_keys=["group"])
    assert set(stats["group"]) == {"A", "B"}
    assert stats.loc[stats["group"] == "A", "n_samples"].iloc[0] == 2


def test_compute_ratio_table_groups():
    ratios = pd.DataFrame(
        {
            "spectrum_id": ["s1", "s2", "s3", "s4"],
            "ratio_1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    meta = pd.DataFrame({"spectrum_id": ["s1", "s2", "s3", "s4"], "group": ["A", "A", "B", "B"]})
    tbl = compute_ratio_table(ratios, metadata=meta, group_keys=["group"])
    assert set(tbl["group"]) == {"A", "B"}
    assert "ratio_1" in set(tbl["ratio_name"])
    assert tbl.loc[tbl["group"] == "A", "n"].iloc[0] == 2
