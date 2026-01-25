"""
High-level evaluator integration for trust and uncertainty quantification.

Provides evaluator classes that integrate trust artifacts with FoodSpec's
evaluation pipeline and registry system.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from foodspec.evaluation.artifact_registry import ArtifactRegistry
from foodspec.trust.conformal import MondrianConformalClassifier, ConformalPredictionResult
from foodspec.trust.abstain import evaluate_abstention, AbstentionResult
from foodspec.trust.calibration import (
    TemperatureScaler,
    IsotonicCalibrator,
    expected_calibration_error,
)


@dataclass
class TrustEvaluationResult:
    """Aggregated trust and uncertainty evaluation results."""
    
    timestamp: str
    model_name: str
    
    # Conformal prediction metrics
    conformal_coverage: float
    conformal_set_size_mean: float
    conformal_set_size_median: float
    conformal_set_size_max: int
    per_bin_coverage: Dict[str, float]  # bin_id -> coverage
    
    # Calibration metrics
    ece: float
    temperature_scale: Optional[float]
    isotonic_applied: bool
    
    # Abstention metrics
    abstention_rate: float
    accuracy_non_abstained: Optional[float]
    accuracy_abstained: Optional[float]
    
    # Coverage under abstention
    coverage_under_abstention: float
    efficiency_gain: Optional[float]  # coverage_u_a / coverage_baseline
    
    # Group-aware metrics
    group_metrics: Dict[str, Dict[str, float]]  # group_id -> {metric -> value}


class TrustEvaluator:
    """High-level evaluator for trust and uncertainty quantification."""
    
    def __init__(
        self,
        model,
        artifact_registry: ArtifactRegistry,
        target_coverage: float = 0.9,
        abstention_threshold: float = 0.7,
        calibration_method: str = "temperature",
        random_state: int = 42,
    ):
        """
        Initialize trust evaluator.
        
        Args:
            model: Fitted classifier with predict_proba()
            artifact_registry: ArtifactRegistry instance for saving artifacts
            target_coverage: Target coverage for conformal prediction
            abstention_threshold: Confidence threshold for abstention
            calibration_method: "temperature", "isotonic", or None
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.artifact_registry = artifact_registry
        self.target_coverage = target_coverage
        self.abstention_threshold = abstention_threshold
        self.calibration_method = calibration_method
        self.random_state = random_state
        
        self._conformal: Optional[MondrianConformalClassifier] = None
        self._calibrator: Optional[Any] = None
        self._conformal_fitted = False
        self._calibrator_fitted = False
    
    def fit_conformal(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        bins_cal: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit conformal prediction on calibration data (disjoint from training).
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            bins_cal: Bin indices for Mondrian conditioning (optional)
        """
        self._conformal = MondrianConformalClassifier(
            self.model,
            target_coverage=self.target_coverage,
        )
        # Model already fitted, skip fit step
        self._conformal._fitted = True
        self._conformal._n_classes = len(np.unique(y_cal))
        
        self._conformal.calibrate(X_cal, y_cal, bins=bins_cal)
        self._conformal_fitted = True
    
    def fit_calibration(
        self,
        y_cal: np.ndarray,
        proba_cal: np.ndarray,
    ) -> None:
        """
        Fit probability calibrator on calibration set.
        
        Args:
            y_cal: Calibration labels
            proba_cal: Calibration predicted probabilities
        """
        if self.calibration_method == "temperature":
            self._calibrator = TemperatureScaler()
        elif self.calibration_method == "isotonic":
            self._calibrator = IsotonicCalibrator()
        elif self.calibration_method is None:
            return
        else:
            raise ValueError(f"Unknown calibration_method: {self.calibration_method}")
        
        self._calibrator.fit(y_cal, proba_cal)
        self._calibrator_fitted = True
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        bins_test: Optional[np.ndarray] = None,
        batch_ids: Optional[np.ndarray] = None,
        group_col: Optional[str] = None,
        df_test: Optional[pd.DataFrame] = None,
        model_name: str = "unknown",
    ) -> TrustEvaluationResult:
        """
        Comprehensive trust evaluation on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            bins_test: Bin indices matching bins_cal (optional)
            batch_ids: Batch/group identifiers for group-aware metrics
            group_col: Column name in df_test for grouping
            df_test: DataFrame with metadata (required if group_col specified)
            model_name: Model identifier
        
        Returns:
            TrustEvaluationResult with all metrics
        """
        if not self._conformal_fitted:
            raise RuntimeError("Conformal prediction not fitted. Call fit_conformal().")
        
        proba_test = self.model.predict_proba(X_test)
        
        # Apply calibration if fitted
        if self._calibrator_fitted and self._calibrator is not None:
            proba_test_cal = self._calibrator.predict(proba_test)
        else:
            proba_test_cal = proba_test
        
        # Conformal prediction
        cp_result = self._conformal.predict_sets(
            X_test,
            bins=bins_test,
            y_true=y_test,
        )
        
        # Calibration error
        ece = expected_calibration_error(y_test, proba_test_cal)
        
        # Abstention
        abstain_result = evaluate_abstention(
            proba_test_cal,
            y_test,
            threshold=self.abstention_threshold,
            prediction_sets=cp_result.prediction_sets,
        )
        
        # Group-aware metrics
        group_metrics = {}
        if batch_ids is not None:
            group_metrics = self._compute_group_metrics(
                cp_result,
                y_test,
                proba_test_cal,
                batch_ids,
            )
        elif group_col is not None and df_test is not None:
            batch_ids = df_test[group_col].values
            group_metrics = self._compute_group_metrics(
                cp_result,
                y_test,
                proba_test_cal,
                batch_ids,
            )
        
        # Efficiency gain
        efficiency_gain = None
        if abstain_result.coverage is not None and cp_result.coverage is not None:
            if cp_result.coverage > 0:
                efficiency_gain = abstain_result.coverage / cp_result.coverage
        
        result = TrustEvaluationResult(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            conformal_coverage=cp_result.coverage or 0.0,
            conformal_set_size_mean=float(np.mean(cp_result.set_sizes)),
            conformal_set_size_median=float(np.median(cp_result.set_sizes)),
            conformal_set_size_max=int(np.max(cp_result.set_sizes)),
            per_bin_coverage=cp_result.per_bin_coverage,
            ece=ece,
            temperature_scale=(
                float(self._calibrator.temperature)
                if isinstance(self._calibrator, TemperatureScaler)
                else None
            ),
            isotonic_applied=isinstance(self._calibrator, IsotonicCalibrator),
            abstention_rate=abstain_result.abstention_rate,
            accuracy_non_abstained=abstain_result.accuracy_non_abstained,
            accuracy_abstained=abstain_result.accuracy_abstained,
            coverage_under_abstention=abstain_result.coverage or 0.0,
            efficiency_gain=efficiency_gain,
            group_metrics=group_metrics,
        )
        
        return result
    
    def _compute_group_metrics(
        self,
        cp_result: ConformalPredictionResult,
        y_test: np.ndarray,
        proba_test: np.ndarray,
        batch_ids: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-group coverage and abstention metrics."""
        group_metrics = {}
        
        for group_id in np.unique(batch_ids):
            mask = batch_ids == group_id
            y_group = y_test[mask]
            proba_group = proba_test[mask]
            sets_group = [cp_result.prediction_sets[i] for i in np.where(mask)[0]]
            
            # Per-group coverage
            coverage = np.mean([
                int(y_group[i] in sets_group[i])
                for i in range(len(y_group))
            ])
            
            # Per-group abstention
            abstain_res = evaluate_abstention(
                proba_group,
                y_group,
                threshold=self.abstention_threshold,
            )
            
            group_metrics[str(group_id)] = {
                "coverage": coverage,
                "set_size_mean": float(np.mean([len(s) for s in sets_group])),
                "abstention_rate": abstain_res.abstention_rate,
                "accuracy_non_abstained": (
                    abstain_res.accuracy_non_abstained or 0.0
                ),
            }
        
        return group_metrics
    
    def save_artifacts(
        self,
        result: TrustEvaluationResult,
        prediction_sets: List[List[int]],
        set_sizes: List[int],
        abstention_mask: np.ndarray,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, str]:
        """
        Save all trust artifacts to registry and disk.
        
        Args:
            result: TrustEvaluationResult object
            prediction_sets: Conformal prediction sets for each test sample
            set_sizes: Size of each prediction set
            abstention_mask: Boolean mask of abstained samples
            output_dir: Optional output directory for serialization
        
        Returns:
            Dictionary mapping artifact names to their registry keys
        """
        artifacts = {}
        
        # Save evaluation result
        result_dict = asdict(result)
        result_key = self.artifact_registry.register(
            name=f"trust_eval_{result.model_name}",
            artifact_type="trust_evaluation",
            content=result_dict,
            metadata={
                "timestamp": result.timestamp,
                "model": result.model_name,
            },
        )
        artifacts["evaluation_result"] = result_key
        
        # Save prediction sets
        sets_df = pd.DataFrame({
            "sample_idx": range(len(prediction_sets)),
            "prediction_set": [str(s) for s in prediction_sets],
            "set_size": set_sizes,
        })
        sets_key = self.artifact_registry.register(
            name=f"prediction_sets_{result.model_name}",
            artifact_type="prediction_sets",
            content=sets_df.to_dict(orient="records"),
            metadata={"model": result.model_name},
        )
        artifacts["prediction_sets"] = sets_key
        
        # Save abstention summary
        abstain_df = pd.DataFrame({
            "sample_idx": range(len(abstention_mask)),
            "abstained": abstention_mask,
        })
        abstain_key = self.artifact_registry.register(
            name=f"abstention_{result.model_name}",
            artifact_type="abstention",
            content=abstain_df.to_dict(orient="records"),
            metadata={"model": result.model_name},
        )
        artifacts["abstention"] = abstain_key
        
        # Save to disk if directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "trust_eval.json", "w") as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            sets_df.to_csv(output_dir / "prediction_sets.csv", index=False)
            abstain_df.to_csv(output_dir / "abstention.csv", index=False)
        
        return artifacts
    
    def report(self, result: TrustEvaluationResult) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "TRUST & UNCERTAINTY QUANTIFICATION REPORT",
            "=" * 70,
            f"\nModel: {result.model_name}",
            f"Timestamp: {result.timestamp}",
            
            "\n--- CONFORMAL PREDICTION ---",
            f"Target Coverage: {self.target_coverage:.1%}",
            f"Achieved Coverage: {result.conformal_coverage:.1%}",
            f"Mean Set Size: {result.conformal_set_size_mean:.2f}",
            f"Median Set Size: {result.conformal_set_size_median:.2f}",
            f"Max Set Size: {result.conformal_set_size_max}",
            
            "\n--- CALIBRATION ---",
            f"Calibration Method: {self.calibration_method}",
            f"ECE: {result.ece:.4f}",
            (
                f"Temperature Scale: {result.temperature_scale:.4f}"
                if result.temperature_scale else "N/A"
            ),
            f"Isotonic: {result.isotonic_applied}",
            
            "\n--- ABSTENTION ---",
            f"Abstention Threshold: {self.abstention_threshold:.1%}",
            f"Abstention Rate: {result.abstention_rate:.1%}",
            (
                f"Accuracy (non-abstained): {result.accuracy_non_abstained:.1%}"
                if result.accuracy_non_abstained is not None else "N/A"
            ),
            (
                f"Accuracy (abstained): {result.accuracy_abstained:.1%}"
                if result.accuracy_abstained is not None else "N/A"
            ),
            
            "\n--- COMBINED METRICS ---",
            f"Coverage under Abstention: {result.coverage_under_abstention:.1%}",
            (
                f"Efficiency Gain: {result.efficiency_gain:.2f}x"
                if result.efficiency_gain else "N/A"
            ),
        ]
        
        # Add per-bin coverage if available
        if result.per_bin_coverage:
            lines.append("\n--- PER-BIN COVERAGE ---")
            for bin_id, cov in result.per_bin_coverage.items():
                lines.append(f"  Bin {bin_id}: {cov:.1%}")
        
        # Add group metrics if available
        if result.group_metrics:
            lines.append("\n--- GROUP-AWARE METRICS ---")
            for group_id, metrics in result.group_metrics.items():
                lines.append(f"\n  Group {group_id}:")
                for metric_name, value in metrics.items():
                    if "rate" in metric_name.lower():
                        lines.append(f"    {metric_name}: {value:.1%}")
                    else:
                        lines.append(f"    {metric_name}: {value:.4f}")
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
