from foodspec.features.bands import integrate_bands
from foodspec.features.fingerprint import (
    correlation_similarity_matrix,
    cosine_similarity_matrix,
)
from foodspec.features.peak_stats import compute_peak_stats, compute_ratio_table
from foodspec.features.peaks import PeakFeatureExtractor, detect_peaks
from foodspec.features.ratios import RatioFeatureGenerator, compute_ratios
from foodspec.features.specs import FeatureEngine, FeatureSpec
from foodspec.features.metrics import (
    feature_cv,
    feature_stability_by_group,
    discriminative_power,
    robustness_vs_variations,
)
from foodspec.features.interpretation import (
    ChemicalMeaning,
    DEFAULT_CHEMICAL_LIBRARY,
    find_chemical_meanings,
    explain_feature_spec,
    explain_feature_set,
)

__all__ = [
    "integrate_bands",
    "compute_ratios",
    "RatioFeatureGenerator",
    "detect_peaks",
    "PeakFeatureExtractor",
    "cosine_similarity_matrix",
    "correlation_similarity_matrix",
    "compute_peak_stats",
    "compute_ratio_table",
    "FeatureSpec",
    "FeatureEngine",
    "feature_cv",
    "feature_stability_by_group",
    "discriminative_power",
    "robustness_vs_variations",
    "ChemicalMeaning",
    "DEFAULT_CHEMICAL_LIBRARY",
    "find_chemical_meanings",
    "explain_feature_spec",
    "explain_feature_set",
]
