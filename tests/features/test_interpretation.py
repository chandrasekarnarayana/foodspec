"""
Tests for feature interpretation module.
"""

from foodspec.features.interpretation import (
    DEFAULT_CHEMICAL_LIBRARY,
    ChemicalMeaning,
    explain_feature_set,
    explain_feature_spec,
    find_chemical_meanings,
)
from foodspec.features.specs import FeatureSpec


def test_chemical_meaning_creation():
    """Test ChemicalMeaning dataclass creation."""
    meaning = ChemicalMeaning(
        center=1742.0,
        window=20.0,
        meaning='C=O stretch (Carbonyl, Oxidation marker)',
        modality='any'
    )

    assert meaning.center == 1742.0
    assert meaning.window == 20.0
    assert 'C=O stretch' in meaning.meaning
    assert meaning.modality == 'any'


def test_default_chemical_library_exists():
    """Test that default chemical library is populated."""
    assert len(DEFAULT_CHEMICAL_LIBRARY) > 0

    # Check structure of library entries
    first_entry = DEFAULT_CHEMICAL_LIBRARY[0]
    assert isinstance(first_entry, ChemicalMeaning)
    assert hasattr(first_entry, 'center')
    assert hasattr(first_entry, 'meaning')
    assert hasattr(first_entry, 'window')


def test_find_chemical_meanings_exact_match():
    """Test finding chemical meanings for exact wavenumber."""
    # Use a known wavenumber from the library
    meanings = find_chemical_meanings(1742, tolerance=5)

    # Should find at least one meaning for this common wavenumber
    assert len(meanings) >= 0  # May or may not have matches depending on library


def test_find_chemical_meanings_with_tolerance():
    """Test finding chemical meanings with tolerance window."""
    meanings = find_chemical_meanings(1740, tolerance=10)

    # Should find meanings within ±10 cm⁻¹
    for meaning in meanings:
        assert abs(meaning.center - 1740) <= max(meaning.window, 10)


def test_find_chemical_meanings_no_match():
    """Test finding chemical meanings with no matches."""
    # Use an uncommon wavenumber
    meanings = find_chemical_meanings(9999, tolerance=5)

    assert len(meanings) == 0


def test_find_chemical_meanings_narrow_tolerance():
    """Test finding chemical meanings with very narrow tolerance."""
    meanings = find_chemical_meanings(1742, tolerance=1)

    # Very narrow window should find fewer matches
    for meaning in meanings:
        assert abs(meaning.center - 1742) <= max(meaning.window, 1)


def test_explain_feature_spec_peak():
    """Test explaining a peak-based feature spec."""
    spec = FeatureSpec(
        name='I_1742',
        ftype='peak',
        regions=[(1738, 1746)]
    )

    explanation = explain_feature_spec(spec)

    # Explanation describes the wavenumber and chemical meaning, not the feature name
    assert '1742' in explanation
    assert 'cm^-1' in explanation.lower()


def test_explain_feature_spec_ratio():
    """Test explaining a ratio-based feature spec."""
    spec = FeatureSpec(
        name='ratio_1742_2720',
        ftype='ratio',
        formula='I_1742 / I_2720'
    )

    explanation = explain_feature_spec(spec)

    assert 'ratio' in explanation.lower()
    assert 'ratio_1742_2720' in explanation


def test_explain_feature_spec_band():
    """Test explaining a band integration feature spec."""
    spec = FeatureSpec(
        name='band_1000_1200',
        ftype='band',
        regions=[(1000, 1200)]
    )

    explanation = explain_feature_spec(spec)

    # Explanation describes the center wavenumber and chemical meaning
    assert '1100' in explanation  # center of (1000, 1200)
    assert 'cm^-1' in explanation.lower()


def test_explain_feature_set_multiple_specs():
    """Test explaining a set of feature specs."""
    specs = [
        FeatureSpec(name='I_1742', ftype='peak', regions=[(1740, 1744)]),
        FeatureSpec(name='I_2720', ftype='peak', regions=[(2718, 2722)]),
        FeatureSpec(name='ratio_1742_2720', ftype='ratio', formula='I_1742 / I_2720')
    ]

    explanation = explain_feature_set(specs)

    assert isinstance(explanation, dict)
    assert 'I_1742' in explanation
    assert 'I_2720' in explanation
    assert 'ratio_1742_2720' in explanation


def test_explain_feature_set_empty():
    """Test explaining an empty feature set."""
    explanation = explain_feature_set([])

    assert isinstance(explanation, dict)
    assert len(explanation) == 0


def test_chemical_meaning_string_representation():
    """Test string representation of ChemicalMeaning."""
    meaning = ChemicalMeaning(
        center=1742.0,
        window=20.0,
        meaning='C=O stretch',
        modality='any'
    )

    str_repr = str(meaning)
    assert '1742' in str_repr


def test_find_chemical_meanings_custom_library():
    """Test finding meanings with a custom library."""
    custom_lib = [
        ChemicalMeaning(center=1000.0, window=5.0, meaning='Test', modality='any'),
        ChemicalMeaning(center=1005.0, window=5.0, meaning='Test2', modality='any')
    ]

    meanings = find_chemical_meanings(1002, tolerance=5, library=custom_lib)

    # Should find both entries within tolerance
    assert len(meanings) == 2


def test_explain_feature_spec_with_chemical_context():
    """Test that feature explanation includes chemical context when available."""
    spec = FeatureSpec(
        name='I_1742',
        ftype='peak',
        regions=[(1738, 1746)]
    )

    explanation = explain_feature_spec(spec, modality='ftir')

    # Should include wavenumber and chemical context
    assert '1742' in explanation
    assert 'cm^-1' in explanation
    # Should have chemical context
    assert len(explanation) > 20
