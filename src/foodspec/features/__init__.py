from foodspec.features.bands import integrate_bands
from foodspec.features.fingerprint import (
    correlation_similarity_matrix,
    cosine_similarity_matrix,
)
from foodspec.features.peak_stats import compute_peak_stats, compute_ratio_table
from foodspec.features.peaks import PeakFeatureExtractor, detect_peaks
from foodspec.features.ratios import RatioFeatureGenerator, compute_ratios

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
]
