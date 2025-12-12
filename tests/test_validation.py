import pandas as pd

from foodspec.validation import validate_dataset


def test_validation_missing_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    diag = validate_dataset(df, required_cols=["c"])
    assert diag["errors"]
    assert "c" in diag["errors"][0]


def test_validation_constant_and_class_warning():
    df = pd.DataFrame({"oil_type": ["A", "A"], "feat": [1, 1], "b": [1, 2]})
    diag = validate_dataset(df, class_col="oil_type", min_classes=2)
    assert any("constant" in w for w in diag["warnings"])
    assert any("discrimination limited" in w for w in diag["warnings"])
