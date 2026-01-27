"""Diagnostics mixin for FoodSpec API - dataset quality assessment."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class FoodSpecDiagnosticsMixin:
    """Mixin class providing dataset diagnostic capabilities for FoodSpec."""

    def summarize_dataset(
        self,
        label_column: Optional[str] = None,
        required_metadata_columns: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive dataset summary for at-a-glance quality assessment.

        **Returns:**
        - class_distribution (if label_column provided)
        - spectral_quality (SNR, range, NaN/inf counts)
        - metadata_completeness
        - dataset_info (n_samples, n_wavenumbers, modality)

        See foodspec.core.summary module docstring for details.

        Parameters
        ----------
        label_column : str, optional
            Column with class labels.
        required_metadata_columns : list of str, optional
            Metadata columns that must be present.

        Returns
        -------
        summary : dict
            Comprehensive dataset summary.
        """
        from foodspec.core.summary import summarize_dataset

        summary = summarize_dataset(
            self.data,
            label_column=label_column,
            required_metadata_columns=required_metadata_columns,
        )

        # Record summary metrics
        self.bundle.add_metrics("dataset_summary", summary)

        return summary

    def check_class_balance(
        self,
        label_column: str,
        severe_threshold: float = 10.0,
        min_samples_per_class: int = 20,
    ) -> Dict[str, Any]:
        """
        Check class balance and flag severe imbalance.

        **Returns:**
        - samples_per_class, imbalance_ratio, severe_imbalance flag
        - undersized_classes, recommended_action

        See foodspec.qc.dataset_qc module docstring for details.

        Parameters
        ----------
        label_column : str
            Column with class labels.
        severe_threshold : float, default=10.0
            Imbalance ratio above which to flag as severe.
        min_samples_per_class : int, default=20
            Minimum recommended samples per class.

        Returns
        -------
        metrics : dict
            Class balance diagnostics.
        """
        from foodspec.qc.dataset_qc import check_class_balance

        balance = check_class_balance(
            self.data.metadata,
            label_column,
            severe_threshold=severe_threshold,
            min_samples_per_class=min_samples_per_class,
        )

        self.bundle.add_metrics("class_balance", balance)

        return balance

    def assess_replicate_consistency(
        self,
        replicate_column: str,
        technical_cv_threshold: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Compute coefficient of variation (CV) for replicate groups.

        **Returns:**
        - cv_per_replicate, median_cv, high_variability_replicates

        See foodspec.qc.replicates module docstring for details.

        Parameters
        ----------
        replicate_column : str
            Column defining replicate groups.
        technical_cv_threshold : float, default=10.0
            CV (%) above which to flag as high variability.

        Returns
        -------
        metrics : dict
            Replicate consistency metrics.
        """
        from foodspec.qc.replicates import compute_replicate_consistency

        consistency = compute_replicate_consistency(
            self.data.x,
            self.data.metadata,
            replicate_column,
            technical_cv_threshold=technical_cv_threshold,
        )

        self.bundle.add_metrics("replicate_consistency", consistency)

        return consistency

    def detect_leakage(
        self,
        label_column: str,
        batch_column: Optional[str] = None,
        replicate_column: Optional[str] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Detect data leakage: batch–label correlation and replicate splits.

        **Returns:**
        - batch_label_correlation (Cramér's V)
        - replicate_leakage (risk/detection)
        - overall_risk: 'high', 'moderate', 'low'

        See foodspec.qc.leakage module docstring for details.

        Parameters
        ----------
        label_column : str
            Column with class labels.
        batch_column : str, optional
            Column defining batches.
        replicate_column : str, optional
            Column defining replicate groups.
        train_indices : np.ndarray, optional
            Row indices for training set.
        test_indices : np.ndarray, optional
            Row indices for test set.

        Returns
        -------
        leakage_report : dict
            Comprehensive leakage diagnostics.
        """
        from foodspec.qc.leakage import detect_leakage

        leakage_report = detect_leakage(
            self.data,
            label_column,
            batch_column=batch_column,
            replicate_column=replicate_column,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        self.bundle.add_metrics("leakage_detection", leakage_report)

        return leakage_report

    def compute_readiness_score(
        self,
        label_column: str,
        batch_column: Optional[str] = None,
        replicate_column: Optional[str] = None,
        required_metadata_columns: Optional[list] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive dataset readiness score (0-100).

        **Scoring Dimensions:**
        - Sample size, class balance, replicate consistency
        - Metadata completeness, spectral quality, leakage risk

        **Returns:**
        - overall_score: 0-100
        - dimension_scores: individual dimension scores
        - passed_criteria, failed_criteria
        - recommendation: text guidance

        See foodspec.qc.readiness module docstring for details.

        Parameters
        ----------
        label_column : str
            Column with class labels.
        batch_column : str, optional
            Column defining batches.
        replicate_column : str, optional
            Column defining replicate groups.
        required_metadata_columns : list of str, optional
            Metadata columns that must be complete.
        weights : dict, optional
            Custom weights for scoring dimensions.

        Returns
        -------
        score_report : dict
            Readiness score report.
        """
        from foodspec.qc.readiness import compute_readiness_score

        score_report = compute_readiness_score(
            self.data,
            label_column,
            batch_column=batch_column,
            replicate_column=replicate_column,
            required_metadata_columns=required_metadata_columns,
            weights=weights,
        )

        self.bundle.add_metrics("readiness_score", score_report)

        return score_report
