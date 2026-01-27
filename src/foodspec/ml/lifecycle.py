from __future__ import annotations

"""Model lifecycle management: aging, sunset rules, and performance tracking.

Tracks model performance over time and automates retirement decisions.
"""


from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats


class ModelState(Enum):
    """Model lifecycle states."""

    DEVELOPMENT = "development"  # Under construction
    CANDIDATE = "candidate"  # Ready for testing
    CHAMPION = "champion"  # Current production model
    CHALLENGER = "challenger"  # Competing with champion
    RETIRED = "retired"  # Sunset/deprecated
    ARCHIVED = "archived"  # Historical reference only


@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement.

    Attributes
    ----------
    timestamp : datetime
        When measurement was taken.
    metric_name : str
        Performance metric (e.g., "accuracy", "auc", "f1").
    metric_value : float
        Measured performance.
    n_samples : int
        Number of samples evaluated.
    metadata : Dict
        Additional context (batch_id, deployment_version, etc.).
    """

    timestamp: datetime
    metric_name: str
    metric_value: float
    n_samples: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class ModelAgingScore:
    """Model aging/degradation assessment.

    Attributes
    ----------
    age_days : float
        Days since model deployment.
    performance_decay : float
        Fractional performance loss since baseline (0=no decay, 1=total loss).
    decay_rate : float
        Performance units lost per day.
    trend_pvalue : float
        Statistical significance of decay trend (low=significant decay).
    recommendation : str
        Action recommendation based on aging analysis.
    is_retired : bool
        True if model should be retired based on sunset rules.
    """

    age_days: float
    performance_decay: float
    decay_rate: float
    trend_pvalue: float
    recommendation: str
    is_retired: bool


@dataclass
class SunsetRule:
    """Automated model retirement criteria.

    Attributes
    ----------
    max_age_days : Optional[float]
        Maximum model age before forced retirement.
    min_performance : Optional[float]
        Minimum acceptable performance (retire if below).
    max_decay_rate : Optional[float]
        Maximum acceptable decay rate (perf units/day).
    min_samples_evaluated : int
        Minimum samples needed to evaluate sunset rules.
    grace_period_days : float
        Days before sunset rules take effect (allow initial burn-in).
    """

    max_age_days: Optional[float] = None
    min_performance: Optional[float] = None
    max_decay_rate: Optional[float] = None
    min_samples_evaluated: int = 100
    grace_period_days: float = 7.0


class ModelLifecycleTracker:
    """Track model performance over time and assess aging.

    Parameters
    ----------
    model_id : str
        Unique model identifier.
    deployment_date : datetime
        When model was deployed to production.
    baseline_performance : float
        Initial/reference performance metric.
    metric_name : str
        Performance metric being tracked (e.g., "accuracy").
    sunset_rule : Optional[SunsetRule]
        Automated retirement criteria.
    """

    def __init__(
        self,
        model_id: str,
        deployment_date: datetime,
        baseline_performance: float,
        metric_name: str = "accuracy",
        sunset_rule: Optional[SunsetRule] = None,
    ):
        self.model_id = model_id
        self.deployment_date = deployment_date
        self.baseline_performance = baseline_performance
        self.metric_name = metric_name
        self.sunset_rule = sunset_rule or SunsetRule()
        self.snapshots: List[PerformanceSnapshot] = []
        self.state = ModelState.CHAMPION

    def record_performance(
        self,
        timestamp: datetime,
        metric_value: float,
        n_samples: int,
        **metadata,
    ) -> None:
        """Record a performance snapshot.

        Parameters
        ----------
        timestamp : datetime
            Measurement time.
        metric_value : float
            Performance metric value.
        n_samples : int
            Number of samples evaluated.
        **metadata
            Additional context.
        """
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            metric_name=self.metric_name,
            metric_value=metric_value,
            n_samples=n_samples,
            metadata=metadata,
        )
        self.snapshots.append(snapshot)

    def compute_aging_score(self, current_time: Optional[datetime] = None) -> ModelAgingScore:
        """Assess model aging and performance decay.

        Parameters
        ----------
        current_time : Optional[datetime]
            Reference time (defaults to now).

        Returns
        -------
        ModelAgingScore
            Aging assessment with retirement recommendation.
        """
        if current_time is None:
            current_time = datetime.now()

        latest_snapshot_time = max(s.timestamp for s in self.snapshots) if self.snapshots else current_time
        ref_time = min(current_time, latest_snapshot_time)
        age_days = (ref_time - self.deployment_date).total_seconds() / 86400.0

        # Insufficient data
        if len(self.snapshots) < 2:
            return ModelAgingScore(
                age_days=age_days,
                performance_decay=0.0,
                decay_rate=0.0,
                trend_pvalue=1.0,
                recommendation="⏳ Insufficient data for aging analysis (need ≥2 snapshots).",
                is_retired=False,
            )

        # Convert snapshots to time series
        df = pd.DataFrame(
            [
                {
                    "timestamp": s.timestamp,
                    "value": s.metric_value,
                    "n_samples": s.n_samples,
                }
                for s in self.snapshots
            ]
        )
        df["days_since_deploy"] = (df["timestamp"] - self.deployment_date).dt.total_seconds() / 86400.0

        # Performance decay (relative to baseline)
        current_perf = df["value"].iloc[-1]
        performance_decay = max(0.0, (self.baseline_performance - current_perf) / self.baseline_performance)

        # Trend analysis (linear regression)
        x = df["days_since_deploy"].values
        y = df["value"].values

        if len(x) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            decay_rate = -float(slope)  # Negative slope = decay
            trend_pvalue = float(p_value)
        else:
            # Simple difference for 2 points
            decay_rate = (df["value"].iloc[0] - df["value"].iloc[-1]) / (x[-1] - x[0] + 1e-10)
            trend_pvalue = 1.0

        # Apply sunset rules
        is_retired = self._check_sunset_rules(
            age_days=age_days,
            current_perf=current_perf,
            decay_rate=decay_rate,
            total_samples=int(df["n_samples"].sum()),
        )

        # Recommendation
        if is_retired:
            recommendation = "🛑 SUNSET: Model should be retired per configured rules."
        elif decay_rate > 0.01 and trend_pvalue < 0.05:
            recommendation = "⚠️ WARNING: Significant performance decay detected. Monitor closely."
        elif performance_decay > 0.1:
            recommendation = "⚡ CAUTION: 10%+ performance loss from baseline. Consider retraining."
        else:
            recommendation = "✅ OK: Model performance stable."

        return ModelAgingScore(
            age_days=age_days,
            performance_decay=performance_decay,
            decay_rate=decay_rate,
            trend_pvalue=trend_pvalue,
            recommendation=recommendation,
            is_retired=is_retired,
        )

    def _check_sunset_rules(
        self,
        age_days: float,
        current_perf: float,
        decay_rate: float,
        total_samples: int,
    ) -> bool:
        """Check if model meets retirement criteria.

        Returns
        -------
        bool
            True if model should be retired.
        """
        rule = self.sunset_rule

        # Grace period bypass
        if age_days < rule.grace_period_days:
            return False

        # Insufficient samples bypass
        if total_samples < rule.min_samples_evaluated:
            return False

        # Age limit
        if rule.max_age_days is not None and age_days > rule.max_age_days:
            return True

        # Performance floor
        if rule.min_performance is not None and current_perf < rule.min_performance:
            return True

        # Decay rate ceiling
        if rule.max_decay_rate is not None and decay_rate > rule.max_decay_rate:
            return True

        return False

    def get_performance_history(self) -> pd.DataFrame:
        """Export performance history as DataFrame.

        Returns
        -------
        pd.DataFrame
            Time series of performance snapshots.
        """
        if not self.snapshots:
            return pd.DataFrame()

        records = []
        for s in self.snapshots:
            record = {
                "timestamp": s.timestamp,
                "metric_name": s.metric_name,
                "metric_value": s.metric_value,
                "n_samples": s.n_samples,
            }
            record.update(s.metadata)
            records.append(record)

        return pd.DataFrame(records)


__all__ = [
    "ModelState",
    "PerformanceSnapshot",
    "ModelAgingScore",
    "SunsetRule",
    "ModelLifecycleTracker",
]
