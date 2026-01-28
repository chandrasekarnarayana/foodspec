"""Streaming monitoring API for real-time QC and drift tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from foodspec.data_objects.spectra_set import FoodSpectrumSet
from foodspec.qc.engine import compute_health_scores, detect_drift, detect_outliers


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class MonitoringEvent:
    timestamp: str
    health_mean: float
    outlier_rate: float
    drift_score: float
    alerts: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


class StreamingMonitor:
    """Stream batches of spectra through QC and drift checks."""

    def __init__(
        self,
        *,
        reference: Optional[np.ndarray] = None,
        wavenumbers: Optional[np.ndarray] = None,
        drift_method: str = "pca_cusum",
        outlier_method: str = "robust_z",
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.reference = np.asarray(reference) if reference is not None else None
        self.wavenumbers = np.asarray(wavenumbers) if wavenumbers is not None else None
        self.drift_method = drift_method
        self.outlier_method = outlier_method
        self.thresholds = thresholds or {
            "health_min": 0.6,
            "outlier_rate": 0.1,
            "drift_score": 1.0,
        }
        self.history: List[MonitoringEvent] = []

    def update(
        self, X: np.ndarray, *, metadata: Optional[Any] = None, timestamp: Optional[str] = None
    ) -> MonitoringEvent:
        """Process a new batch of spectra and return monitoring event."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if self.wavenumbers is None:
            self.wavenumbers = np.arange(X.shape[1])
        ds = FoodSpectrumSet(x=X, wavenumbers=self.wavenumbers, metadata=metadata, modality="unknown")

        health = compute_health_scores(ds)
        outliers = detect_outliers(ds, method=self.outlier_method)
        drift = detect_drift(ds, reference=self.reference, method=self.drift_method)

        outlier_rate = float(np.mean(outliers.labels))
        alerts: List[str] = []
        if health.aggregates.get("health_mean", 1.0) < self.thresholds.get("health_min", 0.6):
            alerts.append("low_health")
        if outlier_rate > self.thresholds.get("outlier_rate", 0.1):
            alerts.append("high_outlier_rate")
        if drift.drift_score > self.thresholds.get("drift_score", 1.0):
            alerts.append("drift_detected")

        event = MonitoringEvent(
            timestamp=timestamp or _now_utc(),
            health_mean=health.aggregates.get("health_mean", 0.0),
            outlier_rate=outlier_rate,
            drift_score=drift.drift_score,
            alerts=alerts,
            details={
                "health": health.aggregates,
                "outliers": {"method": outliers.method},
                "drift": {"method": drift.method, "details": drift.details},
            },
        )
        self.history.append(event)
        return event

    def summary(self) -> Dict[str, Any]:
        """Return a summary snapshot of monitoring history."""
        if not self.history:
            return {"events": 0}
        last = self.history[-1]
        return {
            "events": len(self.history),
            "last_timestamp": last.timestamp,
            "last_alerts": last.alerts,
            "last_health_mean": last.health_mean,
            "last_outlier_rate": last.outlier_rate,
            "last_drift_score": last.drift_score,
        }


__all__ = ["MonitoringEvent", "StreamingMonitor"]
