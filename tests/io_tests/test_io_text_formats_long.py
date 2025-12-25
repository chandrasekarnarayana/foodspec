import pandas as pd

from foodspec.io import text_formats


def test_read_csv_table_long(tmp_path):
    csv = tmp_path / "long.csv"
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "wavenumber": [1000, 1002, 1004, 1000, 1002, 1004],
            "intensity": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "oil_type": ["A", "A", "A", "B", "B", "B"],
        }
    )
    df.to_csv(csv, index=False)
    spectra = text_formats.read_csv_table(csv, format="long", label_column="oil_type")
    assert spectra.x.shape == (2, 3)
    assert list(spectra.metadata["oil_type"]) == ["A", "B"]
