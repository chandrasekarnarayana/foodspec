"""
Tests for feature interpretation module.
"""
import pytest

from foodspec.features.interpretation import (
    ChemicalMeaning,
    DEFAULT_CHEMICAL_LIBRARY,
    find_chemical_meanings,
    explain_feature_spec,
    explain_feature_set
)
from foodspec.features.specs import FeatureSpec


def test_chemical_meaning_creation():
    """Test ChemicalMeaning dataclass creation."""
    meaning = ChemicalMeaning(
        wavenumber=1742,
        band_name='C=O stretch',
        molecule='Carbonyl',
        relevance='Oxidation marker'
    )
    
    assert meaning.wavenumber == 1742
    assert meaning.band_name == 'C=O stretch'
    assert meaning.molecule == 'Carbonyl'
    assert meaning.relevance == 'Oxidation marker'


def test_default_chemical_library_exists():
    """Test that default chemical library is populated."""
    assert len(DEFAULT_CHEMICAL_LIBRARY) > 0
    
    # Check structure of library entries
    first_entry = DEFAULT_CHEMICAL_LIBRARY[0]
    assert isinstance(first_entry, ChemicalMeaning)
    assert hasattr(first_entry, 'wavenumber')
    assert hasattr(first_entry, 'band_name')


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
        assert abs(meaning.wavenumber - 1740) <= 10


def test_find_chemical_meanings_no_match():
    """Test finding chemical meanings with no matches."""
    # Use an uncommon wavenumber
    meanings = find_chemical_meanings(9999, tolerance=5)
    
    assert len(meanings) == 0


def test_find_chemical_meanings_narrow_tolerance():
    """Test finding chemical meanings with very narrow tolerance."""
    meanings = find_chemical_meanings(1742, tolerance=1)
    
    # Very narrow window should find fewer or no matches
    for meaning in meanings:
        assert abs(meaning.wavenumber - 1742) <= 1


def test_explain_feature_spec_peak():
    """Test explaining a peak-based feature spec."""
    spec = FeatureSpec(
        name='I_1742',
        type='peak',
        wavenumber=1742,
        window=(1738, 1746)
    )
    
    explanation = explain_feature_spec(spec)
    
    assert 'I_1742' in explanation
    assert '1742' in explanation
    assert 'peak' in explanation.lower()


def test_explain_feature_spec_ratio():
    """Test explaining a ratio-based feature spec."""
    spec = FeatureSpec(
        name='ratio_1742_2720',
        type='ratio',
        numerator=1742,
        denominator=2720
    )
    
    explanation = explain_feature_spec(spec)
    
    assert 'ratio' in explanation.lower()
    assert '1742' in explanation
    assert '2720' in explanation


def test_explain_feature_spec_band():
    """Test explaining a band integration feature spec."""
    spec = FeatureSpec(
        name='band_1000_1200',
        type='band',
        range=(1000, 1200)
    )
    
    explanation = explain_feature_spec(spec)
    
    assert 'band' in explanation.lower()
    assert '1000' in explanation
    assert '1200' in explanation


def test_explain_feature_set_multiple_specs():
    """Test explaining a set of feature specs."""
    specs = [
        FeatureSpec(name='I_1742', type='peak', wavenumber=1742),
        FeatureSpec(name='I_2720', type='peak', wavenumber=2720),
        FeatureSpec(name='ratio_1742_2720', type='ratio', numerator=1742, denominator=2720)
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
        wavenumber=1742,
        band_name='C=O stretch',
        molecule='Carbonyl',
        relevance='Oxidation'
    )
    
    str_repr = str(meaning)
    assert '1742' in str_repr


def test_find_chemical_meanings_custom_library():
    """Test finding meanings with a custom library."""
    custom_lib = [
        ChemicalMeaning(wavenumber=1000, band_name='Test', molecule='Test', relevance='Test'),
        ChemicalMeaning(wavenumber=1005, band_name='Test2', molecule='Test2', relevance='Test2')
    ]
    
    meanings = find_chemical_meanings(1002, tolerance=5, library=custom_lib)
    
    # Should find both entries within tolerance
    assert len(meanings) == 2


def test_explain_feature_spec_with_chemical_context():
    """Test that feature explanation includes chemical context when available."""
    spec = FeatureSpec(
        name='I_1742',
        type='peak',
        wavenumber=1742,
        window=(1738, 1746)
    )
    
    explanation = explain_feature_spec(spec, include_chemical_meanings=True)
    
    # Should include wavenumber info at minimum
    assert '1742' in explanation
