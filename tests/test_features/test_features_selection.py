import pandas as pd

from foodspec.features.selection import feature_stability_by_group


def test_feature_stability_by_group():
    df = pd.DataFrame(
        {
            "feature1": [1, 1.1, 0.9, 1.0],
            "group": ["a", "a", "b", "b"],
        }
    )
    result = feature_stability_by_group(df[["feature1"]], df["group"])
    assert "feature1" in result.index

