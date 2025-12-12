"""Spectral similarity utilities."""

from __future__ import annotations

import numpy as np

__all__ = ["cosine_similarity_matrix", "correlation_similarity_matrix"]


def cosine_similarity_matrix(X_ref: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between reference and query spectra."""

    X_ref = np.asarray(X_ref, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    ref_norm = np.linalg.norm(X_ref, axis=1, keepdims=True)
    query_norm = np.linalg.norm(X_query, axis=1, keepdims=True)
    ref_norm = np.maximum(ref_norm, np.finfo(float).eps)
    query_norm = np.maximum(query_norm, np.finfo(float).eps)
    sims = (X_ref @ X_query.T) / (ref_norm * query_norm.T)
    return sims


def correlation_similarity_matrix(X_ref: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation similarity matrix."""

    X_ref = np.asarray(X_ref, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    X_ref_centered = X_ref - X_ref.mean(axis=1, keepdims=True)
    X_query_centered = X_query - X_query.mean(axis=1, keepdims=True)
    ref_norm = np.linalg.norm(X_ref_centered, axis=1, keepdims=True)
    query_norm = np.linalg.norm(X_query_centered, axis=1, keepdims=True)
    ref_norm = np.maximum(ref_norm, np.finfo(float).eps)
    query_norm = np.maximum(query_norm, np.finfo(float).eps)
    sims = (X_ref_centered @ X_query_centered.T) / (ref_norm * query_norm.T)
    return sims
