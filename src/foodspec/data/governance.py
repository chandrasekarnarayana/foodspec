"""Metadata Governance: Instrument Profiles, Calibration History, Audit Trails."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

__all__ = ["InstrumentProfile", "CalibrationRecord", "EnvironmentLog", "GovernanceRegistry"]


@dataclass
class InstrumentProfile:
    """Instrument metadata and configuration for reproducibility."""

    instrument_id: str
    instrument_type: str  # e.g., "FTIR", "UV-Vis", "LC-MS"
    manufacturer: str
    model: str
    serial_number: str
    installation_date: str  # ISO format
    last_maintenance: Optional[str] = None
    calibration_status: str = "Valid"  # "Valid", "Expired", "Under Review"
    calibration_interval_days: int = 365
    location: str = ""
    responsible_person: str = ""
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def get_fingerprint(self) -> str:
        """
        Get SHA256 fingerprint for integrity verification.

        Returns
        -------
        fingerprint : str
            SHA256 hash of profile content.
        """
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class CalibrationRecord:
    """Calibration event with traceability."""

    calibration_id: str
    instrument_id: str
    calibration_date: str  # ISO format
    calibration_method: str
    standard_reference: str  # e.g., "NIST SRM 1507"
    performed_by: str
    acceptance_criteria: Dict[str, float]  # e.g., {"rmse": 0.05, "r2": 0.995}
    results: Dict[str, float]  # Actual measurement results
    passed: bool
    valid_until: Optional[str] = None
    calibration_points: int = 0
    frequency_response: Optional[Dict[str, float]] = None  # e.g., for spectroscopy
    notes: str = ""
    audit_trail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def get_certificate(self) -> str:
        """Generate calibration certificate."""
        cert = f"""
CALIBRATION CERTIFICATE
=======================
ID:                     {self.calibration_id}
Instrument:             {self.instrument_id}
Calibration Date:       {self.calibration_date}
Valid Until:            {self.valid_until or "N/A"}

Method:                 {self.calibration_method}
Standard Reference:     {self.standard_reference}
Performed By:           {self.performed_by}

Acceptance Criteria:
{chr(10).join([f"  {k}: {v}" for k, v in self.acceptance_criteria.items()])}

Results:
{chr(10).join([f"  {k}: {v}" for k, v in self.results.items()])}

Status:                 {"PASSED" if self.passed else "FAILED"}
Notes:                  {self.notes}
"""
        return cert.strip()


@dataclass
class EnvironmentLog:
    """Environmental conditions during measurements."""

    log_id: str
    instrument_id: str
    timestamp: str  # ISO format
    temperature_c: float
    humidity_percent: float
    air_pressure_hpa: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def is_within_limits(
        self,
        temp_range: tuple = (20, 25),
        humidity_range: tuple = (30, 70),
        pressure_range: tuple = (950, 1050),
    ) -> bool:
        """
        Check if conditions are within acceptable limits.

        Parameters
        ----------
        temp_range : tuple
            (min, max) temperature in Celsius.
        humidity_range : tuple
            (min, max) relative humidity in percent.
        pressure_range : tuple
            (min, max) air pressure in hPa.

        Returns
        -------
        within_limits : bool
        """
        return (
            temp_range[0] <= self.temperature_c <= temp_range[1]
            and humidity_range[0] <= self.humidity_percent <= humidity_range[1]
            and pressure_range[0] <= self.air_pressure_hpa <= pressure_range[1]
        )


class GovernanceRegistry:
    """Central registry for all governance metadata."""

    def __init__(self):
        """Initialize governance registry."""
        self.instruments: Dict[str, InstrumentProfile] = {}
        self.calibrations: Dict[str, CalibrationRecord] = {}
        self.environment_logs: List[EnvironmentLog] = []
        self.created_at = datetime.utcnow().isoformat()

    def register_instrument(self, profile: InstrumentProfile) -> None:
        """Register new instrument."""
        if profile.instrument_id in self.instruments:
            raise ValueError(f"Instrument {profile.instrument_id} already registered")
        self.instruments[profile.instrument_id] = profile

    def get_instrument(self, instrument_id: str) -> Optional[InstrumentProfile]:
        """Retrieve instrument profile."""
        return self.instruments.get(instrument_id)

    def add_calibration(self, record: CalibrationRecord) -> None:
        """Add calibration record with audit trail."""
        if record.calibration_id in self.calibrations:
            raise ValueError(f"Calibration {record.calibration_id} already exists")

        # Verify instrument exists
        if record.instrument_id not in self.instruments:
            raise ValueError(f"Instrument {record.instrument_id} not registered")

        # Add timestamp to audit trail
        if not record.audit_trail:
            record.audit_trail = {}
        record.audit_trail["registered_at"] = datetime.utcnow().isoformat()

        self.calibrations[record.calibration_id] = record

        # Update instrument calibration status
        if record.passed:
            self.instruments[record.instrument_id].calibration_status = "Valid"
            self.instruments[record.instrument_id].last_maintenance = record.calibration_date

    def get_calibration(self, calibration_id: str) -> Optional[CalibrationRecord]:
        """Retrieve calibration record."""
        return self.calibrations.get(calibration_id)

    def get_instrument_calibrations(self, instrument_id: str) -> List[CalibrationRecord]:
        """Get all calibration records for an instrument."""
        return [cal for cal in self.calibrations.values() if cal.instrument_id == instrument_id]

    def is_instrument_calibrated(self, instrument_id: str) -> bool:
        """Check if instrument has valid calibration."""
        if instrument_id not in self.instruments:
            return False

        profile = self.instruments[instrument_id]
        return profile.calibration_status == "Valid"

    def add_environment_log(self, log: EnvironmentLog) -> None:
        """Log environmental conditions."""
        self.environment_logs.append(log)

    def get_environment_history(self, instrument_id: str) -> List[EnvironmentLog]:
        """Get environmental history for instrument."""
        return [log for log in self.environment_logs if log.instrument_id == instrument_id]

    def validate_measurement_conditions(
        self,
        instrument_id: str,
        measurement_timestamp: str,
        allowed_deviation_minutes: int = 30,
    ) -> Dict[str, Any]:
        """
        Validate that measurement was taken under acceptable conditions.

        Parameters
        ----------
        instrument_id : str
            Instrument identifier.
        measurement_timestamp : str
            Timestamp of measurement (ISO format).
        allowed_deviation_minutes : int
            How far back to look for environment logs.

        Returns
        -------
        validation : dict
            Validation results with acceptable conditions flag.
        """
        from datetime import datetime, timedelta

        meas_dt = datetime.fromisoformat(measurement_timestamp)
        min_time = meas_dt - timedelta(minutes=allowed_deviation_minutes)

        relevant_logs = [
            log
            for log in self.get_environment_history(instrument_id)
            if datetime.fromisoformat(log.timestamp) >= min_time
        ]

        if not relevant_logs:
            return {
                "valid": False,
                "reason": f"No environment logs found within {allowed_deviation_minutes} minutes",
                "n_logs": 0,
            }

        # Check conditions
        all_within_limits = all(log.is_within_limits() for log in relevant_logs)

        avg_temp = sum(log.temperature_c for log in relevant_logs) / len(relevant_logs)
        avg_humidity = sum(log.humidity_percent for log in relevant_logs) / len(relevant_logs)

        return {
            "valid": all_within_limits,
            "n_logs": len(relevant_logs),
            "avg_temperature_c": float(avg_temp),
            "avg_humidity_percent": float(avg_humidity),
            "all_within_limits": bool(all_within_limits),
            "logs": [log.to_dict() for log in relevant_logs],
        }

    def generate_audit_report(self) -> str:
        """Generate comprehensive audit report."""
        report = f"""
GOVERNANCE AUDIT REPORT
=======================
Generated: {datetime.utcnow().isoformat()}
Registry Created: {self.created_at}

INSTRUMENT INVENTORY
-------------------
Total Instruments: {len(self.instruments)}

"""
        for inst_id, profile in self.instruments.items():
            cals = self.get_instrument_calibrations(inst_id)
            latest_cal = cals[-1] if cals else None

            report += f"""
Instrument: {inst_id}
  Type: {profile.instrument_type}
  Serial: {profile.serial_number}
  Calibration Status: {profile.calibration_status}
  Calibrations: {len(cals)}
  Latest: {latest_cal.calibration_date if latest_cal else "None"}
  Passed Latest: {latest_cal.passed if latest_cal else "N/A"}
"""

        report += f"""

CALIBRATION HISTORY
-------------------
Total Records: {len(self.calibrations)}
Passed: {sum(1 for cal in self.calibrations.values() if cal.passed)}
Failed: {sum(1 for cal in self.calibrations.values() if not cal.passed)}

ENVIRONMENT MONITORING
---------------------
Total Log Entries: {len(self.environment_logs)}
"""

        return report

    def export_registry(self, include_history: bool = True) -> Dict[str, Any]:
        """Export full registry as dictionary."""
        return {
            "created_at": self.created_at,
            "exported_at": datetime.utcnow().isoformat(),
            "instruments": [p.to_dict() for p in self.instruments.values()],
            "calibrations": [c.to_dict() for c in self.calibrations.values()] if include_history else [],
            "environment_logs": [e.to_dict() for e in self.environment_logs] if include_history else [],
        }

    def export_registry_json(self, filepath: str, include_history: bool = True) -> None:
        """Export registry to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.export_registry(include_history), f, indent=2, default=str)
