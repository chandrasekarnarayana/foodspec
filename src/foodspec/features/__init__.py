from foodspec.features.bands import integrate_bands
from foodspec.features.ratios import compute_ratios, RatioFeatureGenerator
from foodspec.features.peaks import detect_peaks, PeakFeatureExtractor
from foodspec.features.fingerprint import cosine_similarity_matrix, correlation_similarity_matrix
from foodspec.features.peak_stats import compute_peak_stats, compute_ratio_table

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
