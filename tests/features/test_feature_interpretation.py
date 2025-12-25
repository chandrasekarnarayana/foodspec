from foodspec.features.interpretation import (
    explain_feature_set,
    explain_feature_spec,
    find_chemical_meanings,
)
from foodspec.features.specs import FeatureSpec


def test_meaning_lookup_matches_library_entry():
    matches = find_chemical_meanings(1745.0, modality="ftir")
    assert matches, "Expected at least one chemical meaning match"
    assert any("C=O" in m.meaning for m in matches)


def test_explain_feature_spec_for_peak():
    spec = FeatureSpec(name="carbonyl_peak", ftype="peak", regions=[(1740.0, 1750.0)], constraints={"modality": "ftir"})
    text = explain_feature_spec(spec, direction="increase")
    assert "1745" in text
    assert "C=O" in text


def test_explain_feature_set_handles_ratios():
    spec_band = FeatureSpec(name="band_a", ftype="band", regions=[(1000.0, 1010.0)])
    spec_ratio = FeatureSpec(name="ratio_a", ftype="ratio", formula="band_a / band_a")
    mapping = explain_feature_set([spec_band, spec_ratio])
    assert "ratio" in mapping["ratio_a"].lower()
    assert "band_a" in mapping["ratio_a"]
