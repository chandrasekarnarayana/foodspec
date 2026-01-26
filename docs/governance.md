# Governance & Metadata Management Guide

## Overview

FoodSpec governance system provides comprehensive metadata tracking, instrument management, and audit trails for regulatory compliance. This guide covers best practices for deploying models in controlled environments.

---

## 1. Instrument Management

### Registration

Every analytical instrument must be registered before use:

```python
from foodspec.data.governance import InstrumentProfile, GovernanceRegistry

registry = GovernanceRegistry()

# Register primary instrument
primary_ftir = InstrumentProfile(
    instrument_id="FTIR_LAB_002_PRIMARY",
    instrument_type="FTIR",
    manufacturer="PerkinElmer",
    model="Spectrum Two",
    serial_number="SN2024001234",
    installation_date="2024-01-15",
    calibration_status="Valid",
    calibration_interval_days=365,
    location="Building A, Lab 202",
    responsible_person="Dr. Alice Johnson (alice@company.com)",
    additional_metadata={
        "purchase_date": "2023-12-01",
        "warranty_end": "2025-12-01",
        "maintenance_contract": "Premium",
    }
)

registry.register_instrument(primary_ftir)

# Register backup instrument
backup_ftir = InstrumentProfile(
    instrument_id="FTIR_LAB_002_BACKUP",
    instrument_type="FTIR",
    manufacturer="PerkinElmer",
    model="Spectrum Two",
    serial_number="SN2024005678",
    installation_date="2024-01-15",
    location="Building A, Lab 202",
    responsible_person="Dr. Alice Johnson",
)

registry.register_instrument(backup_ftir)
```

### Instrument Lifecycle Management

```python
# Track instrument history
instruments = registry.get_instrument("FTIR_LAB_002_PRIMARY")

# Check status
print(f"Status: {instruments.calibration_status}")
print(f"Last maintenance: {instruments.last_maintenance}")
print(f"Next calibration due: {datetime.fromisoformat(instruments.installation_date) + timedelta(days=instruments.calibration_interval_days)}")

# List all instruments of type
all_ftirs = [p for p in registry.instruments.values() if p.instrument_type == "FTIR"]
print(f"Total FTIR systems: {len(all_ftirs)}")
```

---

## 2. Calibration Management

### Calibration Records

Every measurement must be traceable to calibration standards:

```python
from foodspec.data.governance import CalibrationRecord
import json

# Perform calibration
calibration = CalibrationRecord(
    calibration_id="CAL_2024_001_FTIR_LAB_002",
    instrument_id="FTIR_LAB_002_PRIMARY",
    calibration_date="2024-01-15T09:30:00",
    calibration_method="Direct comparison to NIST reference materials",
    standard_reference="NIST SRM 1507 (Polystyrene film)",
    performed_by="John Smith (john.smith@company.com)",
    acceptance_criteria={
        "transmittance_error": 0.05,
        "wavenumber_accuracy": 1.0,  # cm⁻¹
        "resolution": 4.0,  # cm⁻¹
        "r_squared": 0.995,
    },
    results={
        "transmittance_error": 0.038,
        "wavenumber_accuracy": 0.8,
        "resolution": 4.0,
        "r_squared": 0.9973,
    },
    passed=True,
    valid_until="2025-01-15",
    calibration_points=20,
    frequency_response={
        "peak_position_4000_cm": {"expected": 4000, "observed": 4000.0},
        "peak_position_2850_cm": {"expected": 2850, "observed": 2849.8},
        "peak_position_1600_cm": {"expected": 1600, "observed": 1599.9},
    },
    notes="Calibration successful. Instrument ready for use.",
    audit_trail={
        "approved_by": "QA_Manager_ID_001",
        "approval_date": "2024-01-15T14:00:00",
    }
)

registry.add_calibration(calibration)

# Generate certificate
print(calibration.get_certificate())

# Export certificate
with open(f"calibration_certificates/{calibration.calibration_id}.pdf", "w") as f:
    f.write(calibration.get_certificate())
```

### Calibration Schedule

```python
# Create calibration schedule
calibration_schedule = {
    "FTIR_LAB_002_PRIMARY": {
        "interval_days": 365,
        "method": "Direct comparison",
        "standard": "NIST SRM 1507",
    },
    "FTIR_LAB_002_BACKUP": {
        "interval_days": 365,
        "method": "Direct comparison",
        "standard": "NIST SRM 1507",
    },
}

# Check if calibrations are due
from datetime import datetime, timedelta

for instrument_id, schedule in calibration_schedule.items():
    instrument = registry.get_instrument(instrument_id)
    calibrations = registry.get_instrument_calibrations(instrument_id)
    
    if calibrations:
        latest_cal = calibrations[-1]
        due_date = datetime.fromisoformat(latest_cal.calibration_date) + timedelta(days=schedule["interval_days"])
        
        if datetime.utcnow() > due_date:
            print(f"⚠️  OVERDUE: {instrument_id} calibration due")
        else:
            days_remaining = (due_date - datetime.utcnow()).days
            print(f"✓ {instrument_id}: {days_remaining} days until next calibration")
```

### Failed Calibrations

```python
# Log failed calibration attempt
failed_cal = CalibrationRecord(
    calibration_id="CAL_2024_002_FTIR_LAB_002_FAILED",
    instrument_id="FTIR_LAB_002_PRIMARY",
    calibration_date="2024-01-20T10:00:00",
    calibration_method="Direct comparison",
    standard_reference="NIST SRM 1507",
    performed_by="John Smith",
    acceptance_criteria={"r_squared": 0.995},
    results={"r_squared": 0.985},  # Failed
    passed=False,
    notes="Instrument out of specification. Service call initiated.",
    audit_trail={"incident_ticket": "INC_2024_001"},
)

registry.add_calibration(failed_cal)

# Update instrument status
instrument = registry.get_instrument("FTIR_LAB_002_PRIMARY")
instrument.calibration_status = "Expired"
instrument.last_maintenance = "2024-01-20"
```

---

## 3. Environmental Monitoring

### Continuous Logging

Track environmental conditions for traceability:

```python
from foodspec.data.governance import EnvironmentLog

# Log conditions before measurement
env_log = EnvironmentLog(
    log_id="ENV_2024_001_FTIR_LAB_002",
    instrument_id="FTIR_LAB_002_PRIMARY",
    timestamp="2024-01-22T14:30:00",
    temperature_c=22.5,
    humidity_percent=45.2,
    air_pressure_hpa=1013.2,
    notes="Normal lab conditions"
)

registry.add_environment_log(env_log)

# Automated logging (pseudo-code)
class EnvironmentSensor:
    def __init__(self, registry, instrument_id):
        self.registry = registry
        self.instrument_id = instrument_id
    
    def log_every_hour(self):
        """Background task to log conditions"""
        while True:
            temp = self.read_temperature()
            humidity = self.read_humidity()
            pressure = self.read_pressure()
            
            log = EnvironmentLog(
                log_id=f"ENV_{uuid.uuid4()}",
                instrument_id=self.instrument_id,
                timestamp=datetime.utcnow().isoformat(),
                temperature_c=temp,
                humidity_percent=humidity,
                air_pressure_hpa=pressure,
            )
            
            self.registry.add_environment_log(log)
            time.sleep(3600)  # Log every hour
```

### Condition Validation

```python
# Validate conditions at time of measurement
validation = registry.validate_measurement_conditions(
    instrument_id="FTIR_LAB_002_PRIMARY",
    measurement_timestamp="2024-01-22T14:35:00",
    allowed_deviation_minutes=30
)

if validation['valid']:
    print(f"✓ Conditions acceptable")
    print(f"  Temperature: {validation['avg_temperature_c']:.1f}°C")
    print(f"  Humidity: {validation['avg_humidity_percent']:.1f}%")
else:
    print(f"✗ REJECT: {validation['reason']}")
    print(f"  Need logs within 30 minutes of measurement")

# Define acceptable ranges per protocol
ACCEPTABLE_CONDITIONS = {
    "temperature_range": (20.0, 25.0),  # °C
    "humidity_range": (30, 70),          # %
    "pressure_range": (950, 1050),       # hPa
}

def validate_with_protocol(validation, protocol_limits):
    """Check against protocol limits"""
    avg_temp = validation['avg_temperature_c']
    avg_hum = validation['avg_humidity_percent']
    
    if not (protocol_limits['temperature_range'][0] <= avg_temp <= protocol_limits['temperature_range'][1]):
        return False, f"Temperature {avg_temp:.1f}°C outside {protocol_limits['temperature_range']}"
    
    if not (protocol_limits['humidity_range'][0] <= avg_hum <= protocol_limits['humidity_range'][1]):
        return False, f"Humidity {avg_hum:.1f}% outside {protocol_limits['humidity_range']}"
    
    return True, "All conditions within protocol limits"

valid, msg = validate_with_protocol(validation, ACCEPTABLE_CONDITIONS)
print(msg)
```

---

## 4. Audit Reports

### Generate Governance Audit

```python
# Generate comprehensive audit report
report = registry.generate_audit_report()
print(report)

# Export audit report
with open("governance_audit_report.txt", "w") as f:
    f.write(report)

# Generate JSON export for audit systems
registry_data = registry.export_registry(include_history=True)

# Save to file
registry.export_registry_json(
    "governance_registry_export_2024-01-22.json",
    include_history=True
)

# Example: Query specific audit information
def audit_query_by_date_range(registry, start_date, end_date):
    """Get all calibrations in date range"""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    relevant_calibrations = [
        cal for cal in registry.calibrations.values()
        if start <= datetime.fromisoformat(cal.calibration_date) <= end
    ]
    
    return relevant_calibrations

cals_jan_2024 = audit_query_by_date_range(
    registry,
    "2024-01-01T00:00:00",
    "2024-01-31T23:59:59"
)
print(f"Found {len(cals_jan_2024)} calibrations in January 2024")
```

### Traceability Reports

```python
# Generate traceability report for specific measurement
def generate_traceability_report(registry, sample_id, measurement_date, instrument_id):
    """Create complete traceability chain"""
    
    instrument = registry.get_instrument(instrument_id)
    calibrations = registry.get_instrument_calibrations(instrument_id)
    
    # Find calibration valid at measurement time
    meas_dt = datetime.fromisoformat(measurement_date)
    valid_calibration = None
    
    for cal in sorted(calibrations, key=lambda x: x.calibration_date, reverse=True):
        cal_dt = datetime.fromisoformat(cal.calibration_date)
        if cal_dt <= meas_dt:
            valid_calibration = cal
            break
    
    # Get environmental conditions
    env_history = registry.get_environment_history(instrument_id)
    meas_logs = [
        log for log in env_history
        if abs((datetime.fromisoformat(log.timestamp) - meas_dt).total_seconds()) < 30*60
    ]
    
    report = f"""
TRACEABILITY REPORT
====================
Sample ID:          {sample_id}
Measurement Date:   {measurement_date}
Instrument:         {instrument_id}

INSTRUMENT INFORMATION
Manufacturer:       {instrument.manufacturer}
Model:              {instrument.model}
Serial Number:      {instrument.serial_number}

CALIBRATION INFORMATION
Calibration ID:     {valid_calibration.calibration_id if valid_calibration else 'N/A'}
Calibration Date:   {valid_calibration.calibration_date if valid_calibration else 'N/A'}
Valid Until:        {valid_calibration.valid_until if valid_calibration else 'N/A'}
Standard Reference: {valid_calibration.standard_reference if valid_calibration else 'N/A'}

ENVIRONMENTAL CONDITIONS
Temperature (°C):   {env_history[-1].temperature_c if env_history else 'N/A'}
Humidity (%):       {env_history[-1].humidity_percent if env_history else 'N/A'}
Pressure (hPa):     {env_history[-1].air_pressure_hpa if env_history else 'N/A'}

STATUS:             {'✓ VALID' if valid_calibration and valid_calibration.passed else '✗ INVALID'}
"""
    return report.strip()

# Generate report
traceability = generate_traceability_report(
    registry,
    "SAMPLE_2024_001",
    "2024-01-22T14:35:00",
    "FTIR_LAB_002_PRIMARY"
)
print(traceability)
```

---

## 5. Data Integrity

### Fingerprinting

```python
# Verify data integrity using fingerprints
profile = registry.get_instrument("FTIR_LAB_002_PRIMARY")
original_fingerprint = profile.get_fingerprint()
print(f"Original fingerprint: {original_fingerprint}")

# Check if profile has been modified
current_fingerprint = profile.get_fingerprint()
if original_fingerprint == current_fingerprint:
    print("✓ Profile integrity verified")
else:
    print("✗ ALERT: Profile has been modified!")

# Store fingerprints for audit
fingerprint_registry = {
    "2024-01-15_registration": original_fingerprint,
    "2024-01-20_checked": current_fingerprint,
}
```

### Version Control

```python
# Track governance changes over time
import json
from datetime import datetime

class GovernanceVersionControl:
    def __init__(self, registry):
        self.registry = registry
        self.versions = []
    
    def snapshot(self, reason="Routine snapshot"):
        """Create versioned snapshot"""
        version = {
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "fingerprint": hashlib.sha256(
                json.dumps(self.registry.export_registry(), default=str).encode()
            ).hexdigest(),
            "n_instruments": len(self.registry.instruments),
            "n_calibrations": len(self.registry.calibrations),
        }
        self.versions.append(version)
        return version
    
    def get_change_history(self):
        """Get history of changes"""
        return self.versions

# Usage
vc = GovernanceVersionControl(registry)

# Snapshot before major change
vc.snapshot("Pre-instrument registration")

# ... make changes ...

# Snapshot after change
vc.snapshot("Post-calibration update")

# Review history
for v in vc.get_change_history():
    print(f"{v['timestamp']}: {v['reason']}")
```

---

## 6. Access Control & Personnel

### Personnel Management

```python
class PersonnelRegistry:
    def __init__(self):
        self.personnel = {}
        self.permissions = {}
    
    def register_personnel(self, person_id, name, email, roles):
        """Register lab personnel"""
        self.personnel[person_id] = {
            "name": name,
            "email": email,
            "roles": roles,
            "registered_date": datetime.utcnow().isoformat(),
        }
    
    def assign_instrument_responsibility(self, person_id, instrument_id):
        """Assign instrument to personnel"""
        if person_id not in self.permissions:
            self.permissions[person_id] = []
        self.permissions[person_id].append(instrument_id)
    
    def can_use_instrument(self, person_id, instrument_id):
        """Check access"""
        return instrument_id in self.permissions.get(person_id, [])
    
    def get_audit_trail_for_instrument(self, instrument_id):
        """Get who has worked on instrument"""
        responsible = [
            p for p, insts in self.permissions.items()
            if instrument_id in insts
        ]
        return responsible

# Usage
personnel = PersonnelRegistry()
personnel.register_personnel(
    "USER_001",
    "Alice Johnson",
    "alice@company.com",
    ["QA Manager", "Instrument Operator"]
)
personnel.assign_instrument_responsibility("USER_001", "FTIR_LAB_002_PRIMARY")

# Verify access
can_access = personnel.can_use_instrument("USER_001", "FTIR_LAB_002_PRIMARY")
print(f"Access granted: {can_access}")
```

---

## 7. Compliance Checklists

### Governance Compliance

```python
governance_compliance = {
    "Instrument Registration": {
        "status": "Complete",
        "instruments_registered": 2,
        "last_checked": "2024-01-22",
    },
    "Calibration Records": {
        "status": "Complete",
        "calibrations_valid": 2,
        "next_calibration_due": "2025-01-15",
    },
    "Environmental Monitoring": {
        "status": "Active",
        "logs_count": 168,
        "coverage": "7 days continuous",
    },
    "Personnel Management": {
        "status": "Complete",
        "authorized_users": 3,
        "last_access_audit": "2024-01-22",
    },
    "Data Integrity": {
        "status": "Verified",
        "fingerprints_match": True,
        "version_control": "Active",
    },
    "Audit Trail": {
        "status": "Complete",
        "retention_days": 2555,  # 7 years
        "exportable": True,
    },
}

# Check compliance
print("GOVERNANCE COMPLIANCE CHECKLIST")
for category, details in governance_compliance.items():
    status_symbol = "✓" if details['status'] in ["Complete", "Active", "Verified"] else "✗"
    print(f"{status_symbol} {category}: {details['status']}")
```

---

## 8. Best Practices Summary

### Do's ✓

- **Register all instruments** before first use
- **Maintain continuous calibration** records with traceability
- **Log environmental conditions** automatically and continuously
- **Verify measurement conditions** before accepting results
- **Generate audit reports** on regular schedule
- **Use fingerprints** for data integrity verification
- **Export full registries** regularly for backup
- **Document all personnel** responsible for instruments
- **Keep 7-year audit trails** minimum
- **Review compliance quarterly**

### Don'ts ✗

- Don't use instruments without current calibration
- Don't accept measurements outside environmental limits
- Don't delete or modify calibration records
- Don't skip personnel registration
- Don't assume "stable conditions" without logging
- Don't ignore calibration expiration dates
- Don't store metadata without backups
- Don't mix instruments between labs without registration
- Don't accept measurements without traceability
- Don't skip audit reviews

---

## References

- ISO/IEC 17025: General requirements for competence of testing and calibration laboratories
- 21 CFR Part 11: Electronic Records; Electronic Signatures
- JCGM 100:2008: Guide to the Expression of Uncertainty in Measurement
- ISO Guide 35: Certification of reference materials
