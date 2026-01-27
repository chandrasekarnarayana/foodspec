from foodspec.features.alignment import (
    CrossCorrelationAligner,
    DynamicTimeWarpingAligner,
    align_spectra,
)
from foodspec.features.bands import extract_band_features, integrate_bands
from foodspec.features.confidence import add_confidence, decision_from_confidence
from foodspec.features.embeddings import pca_embeddings, pls_embeddings
from foodspec.features.fingerprint import (
    correlation_similarity_matrix,
    cosine_similarity_matrix,
)
from foodspec.features.hybrid import combine_feature_tables, extract_features, scale_features
from foodspec.features.interpretation import (
    DEFAULT_CHEMICAL_LIBRARY,
    ChemicalMeaning,
    explain_feature_set,
    explain_feature_spec,
    find_chemical_meanings,
)
from foodspec.features.library import LibraryIndex, similarity_search
from foodspec.features.marker_panel import build_marker_panel, export_marker_panel
from foodspec.features.metrics import (
    discriminative_power,
    feature_cv,
    feature_stability_by_group,
    robustness_vs_variations,
)
from foodspec.features.peak_stats import compute_peak_stats, compute_ratio_table
from foodspec.features.peaks import PeakFeatureExtractor, detect_peaks, extract_peak_features
from foodspec.features.ratios import RatioFeatureGenerator, compute_ratios
from foodspec.features.rq import (
    PeakDefinition,
    RatioDefinition,
    RatioQualityEngine,
    RatioQualityResult,
    RQConfig,
)
from foodspec.features.schema import BandSpec, FeatureConfig, FeatureInfo, PeakSpec, RatioSpec
from foodspec.features.selection import compute_minimal_panel, feature_importance_scores, stability_selection
from foodspec.features.specs import FeatureEngine, FeatureSpec
from foodspec.features.unmixing import NNLSUnmixer, unmix_spectrum

__all__ = [
    # alignment & unmixing
    "CrossCorrelationAligner",
    "DynamicTimeWarpingAligner",
    "align_spectra",
    "NNLSUnmixer",
    "unmix_spectrum",
    # bands
    "integrate_bands",
    "extract_band_features",
    "compute_ratios",
    "RatioFeatureGenerator",
    "detect_peaks",
    "extract_peak_features",
    "PeakFeatureExtractor",
    "pca_embeddings",
    "pls_embeddings",
    "combine_feature_tables",
    "scale_features",
    "extract_features",
    "build_marker_panel",
    "export_marker_panel",
    "BandSpec",
    "PeakSpec",
    "RatioSpec",
    "FeatureConfig",
    "FeatureInfo",
    "cosine_similarity_matrix",
    "correlation_similarity_matrix",
    "compute_peak_stats",
    "compute_ratio_table",
    # library search & confidence
    "LibraryIndex",
    "similarity_search",
    "add_confidence",
    "decision_from_confidence",
    "FeatureSpec",
    "FeatureEngine",
    "feature_cv",
    "feature_stability_by_group",
    "stability_selection",
    "feature_importance_scores",
    "discriminative_power",
    "robustness_vs_variations",
    "ChemicalMeaning",
    "DEFAULT_CHEMICAL_LIBRARY",
    "find_chemical_meanings",
    "explain_feature_spec",
    "explain_feature_set",
    "PeakDefinition",
    "RatioDefinition",
    "RQConfig",
    "RatioQualityEngine",
    "RatioQualityResult",
    "compute_minimal_panel",
]
