"""
Conformal prediction with Mondrian (group-conditional) calibration.

Provides distribution-free uncertainty quantification with per-group
coverage guarantees, enabling reliable abstention in critical applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np


@dataclass
class ConformalPredictionResult:
    """Result of conformal prediction."""
    
    prediction_sets: List[List[int]]
    """Predicted label sets for each sample."""
    
    set_sizes: List[int]
    """Size of each prediction set."""
    
    coverage: float
    """Empirical coverage (fraction where true label in set)."""
    
    per_bin_coverage: Dict[str, float] = field(default_factory=dict)
    """Coverage per group/bin (for Mondrian conditioning)."""
    
    nonconformity_scores: Optional[np.ndarray] = None
    """Nonconformity scores used for calibration."""
    
    threshold: Optional[float] = None
    """Conformal threshold used."""
    
    target_coverage: Optional[float] = None
    """Target coverage level."""


class MondrianConformalClassifier:
    """
    Conformal prediction with Mondrian group conditioning.
    
    Provides rigorous uncertainty sets with:
    - Global coverage guarantee: P(y ∈ Ĉ(x)) ≥ 1 - α
    - Per-group coverage: conditional guarantee for each batch/group
    - Deterministic outputs with seed control
    
    Must be used with pre-fitted classifier and disjoint calibration set.
    """
    
    def __init__(self, model, target_coverage: float = 0.9):
        """
        Initialize conformal classifier.
        
        Args:
            model: Fitted sklearn classifier with predict_proba()
            target_coverage: Target coverage level (0, 1)
        
        Raises:
            ValueError: If target_coverage not in (0, 1)
        """
        if not (0 < target_coverage < 1):
            raise ValueError(f"target_coverage must be in (0, 1), got {target_coverage}")
        
        self.model = model
        self.target_coverage = target_coverage
        
        self._fitted = False
        self._n_classes = None
        self._thresholds: Dict[str, float] = {}
        self._per_bin_thresholds: Dict[str, Dict[str, float]] = {}
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the base model on training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self._fitted = True
        self._n_classes = len(np.unique(y_train))
    
    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        bins: Optional[np.ndarray] = None,
    ) -> None:
        """
        Calibrate conformal threshold on holdout calibration set.
        
        Computes quantile of nonconformity scores to achieve target coverage.
        Implements Mondrian conditioning if bins provided.
        
        Args:
            X_cal: Calibration features (disjoint from training)
            y_cal: Calibration labels
            bins: Optional bin/group indices for Mondrian conditioning
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get predicted probabilities
        proba = self.model.predict_proba(X_cal)
        
        # Compute nonconformity scores (1 - max softmax prob for correct class)
        predictions = np.argmax(proba, axis=1)
        correct_probs = proba[np.arange(len(y_cal)), y_cal]
        nonconformity = 1.0 - correct_probs
        
        # Global threshold
        alpha = 1.0 - self.target_coverage
        n_cal = len(y_cal)
        
        # Compute quantile: ceil((n+1)(1-alpha)) / n
        quantile_idx = int(np.ceil((n_cal + 1) * (1.0 - alpha))) - 1
        quantile_idx = np.clip(quantile_idx, 0, n_cal - 1)
        
        threshold = np.sort(nonconformity)[quantile_idx]
        # Store both legacy and human-readable keys for compatibility
        self._thresholds["global"] = threshold
        self._thresholds["__global__"] = threshold
        
        # Mondrian conditioning per bin
        if bins is not None:
            unique_bins = np.unique(bins)
            for bin_id in unique_bins:
                mask = bins == bin_id
                nc_bin = nonconformity[mask]
                
                quantile_idx_bin = int(np.ceil((mask.sum() + 1) * (1.0 - alpha))) - 1
                quantile_idx_bin = np.clip(quantile_idx_bin, 0, mask.sum() - 1)
                
                threshold_bin = np.sort(nc_bin)[quantile_idx_bin]
                self._thresholds[str(bin_id)] = threshold_bin
    
    def predict_sets(
        self,
        X_test: np.ndarray,
        bins: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
    ) -> ConformalPredictionResult:
        """
        Generate prediction sets with conformal guarantees.
        
        Args:
            X_test: Test features
            bins: Optional bin indices matching calibration bins
            y_true: Optional true labels for coverage computation
        
        Returns:
            ConformalPredictionResult with sets, sizes, and coverage
        
        Raises:
            RuntimeError: If calibration not fitted
        """
        if not self._thresholds:
            raise RuntimeError("Conformal not calibrated. Call calibrate() first.")
        
        proba = self.model.predict_proba(X_test)
        
        # Compute nonconformity scores
        correct_probs = np.max(proba, axis=1)
        nonconformity = 1.0 - correct_probs
        
        # Generate prediction sets
        prediction_sets = []
        set_sizes = []
        
        for i in range(len(X_test)):
            # Choose threshold: use bin-specific if available, else global
            if bins is not None and str(bins[i]) in self._thresholds:
                thresh = self._thresholds[str(bins[i])]
            else:
                thresh = self._thresholds.get("global", self._thresholds.get("__global__", 0.5))
            
            # Prediction set: classes where nonconformity <= threshold
            nc_class = 1.0 - proba[i]
            pred_set = [c for c in range(self._n_classes) if nc_class[c] <= thresh]
            
            # Fallback: always include predicted class
            if not pred_set:
                pred_set = [np.argmax(proba[i])]
            
            prediction_sets.append(pred_set)
            set_sizes.append(len(pred_set))
        
        # Compute coverage if labels provided
        coverage = None
        per_bin_coverage = {}
        
        if y_true is not None:
            correct = sum(
                1 for i, pred_set in enumerate(prediction_sets)
                if y_true[i] in pred_set
            )
            coverage = correct / len(y_true)
            
            # Per-bin coverage
            if bins is not None:
                for bin_id in np.unique(bins):
                    mask = bins == bin_id
                    correct_bin = sum(
                        1 for i in np.where(mask)[0]
                        if y_true[i] in prediction_sets[i]
                    )
                    per_bin_coverage[str(bin_id)] = correct_bin / mask.sum()

            # Global coverage entry for compatibility with callers/tests
            per_bin_coverage["__global__"] = coverage
        
        return ConformalPredictionResult(
            prediction_sets=prediction_sets,
            set_sizes=set_sizes,
            coverage=coverage,
            per_bin_coverage=per_bin_coverage,
            nonconformity_scores=nonconformity,
            threshold=self._thresholds.get("global", self._thresholds.get("__global__")),
            target_coverage=self.target_coverage,
        )


__all__ = ["MondrianConformalClassifier", "ConformalPredictionResult"]
