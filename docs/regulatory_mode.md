# Regulatory Mode Deployment Guide

## Overview

FoodSpec Phase 2 introduces **Regulatory Mode** - a comprehensive, auditable platform for deploying machine learning models in regulated industries (pharma, food, diagnostics). The platform implements GxP principles (Good Experimental/Laboratory/Manufacturing Practice) and CFR 21 Part 11 compliance concepts.

### Key Features

- **Complete Audit Trail**: Every decision logged with full context
- **Uncertainty Quantification**: Bootstrap, quantile regression, and conformal methods
- **Multivariate Drift Detection**: MMD and Wasserstein distance monitoring
- **Decision Policy**: Cost-sensitive optimization with regulatory logging
- **Governance Registry**: Instrument profiles, calibration history, environmental tracking
- **Regulatory Dossier**: Auto-generated PDF/JSON with version locking

---

## 1. Activating Regulatory Mode

### Via Protocol Configuration

```python
from foodspec.protocol.runner import ProtocolRunner
from foodspec.protocol.schema import Protocol

protocol = Protocol(
    # ... standard protocol configuration ...
    regulatory_mode=True,  # Enable regulatory compliance checks
    regulatory_settings={
        "require_uncertainty": True,        # Enforce prediction intervals
        "require_decision_policy": True,    # Enforce policy audit logs
        "require_calibration": True,        # Verify instrument calibration
        "audit_log_path": "/data/audit_logs/",
        "dossier_output": "/data/regulatory_dossiers/",
    }
)

runner = ProtocolRunner(protocol)
runner.execute()
```

### Via CLI

```bash
foodspec protocol run config.yaml \
  --regulatory \
  --audit-log-dir /data/audit_logs \
  --dossier-dir /data/regulatory_dossiers
```

---

## 2. Quality Control (QC) Expansion

### CUSUM Control Charts

Detect small process shifts with cumulative sum charts:

```python
from foodspec.qc.capability import CUSUMChart
import numpy as np

# Initialize with reference data
reference_data = np.random.normal(100, 5, 100)
chart = CUSUMChart(target=100, k=0.5, h=4.77)
chart.initialize(reference_data)

# Monitor incoming data
for measurement in incoming_stream:
    C_pos, C_neg, is_alarm = chart.update(measurement)
    if is_alarm:
        print(f"ALERT: Process drift detected at value {measurement}")
        # Trigger corrective action

# Get run length statistics
stats = chart.get_run_length()
print(f"Run length: {stats['run_length']}")
print(f"Average run length: {stats['avg_run_length']}")

# Visualize
chart.plot()
```

### Process Capability Analysis

Assess process capability with Cp/Cpk indices:

```python
from foodspec.qc.capability import ProcessCapability

# Define specification limits
analyzer = ProcessCapability(
    lower_spec=90.0,
    upper_spec=110.0
)

# Analyze process
measurements = np.random.normal(100, 2, 100)
results = analyzer.analyze(measurements, sample_size=5)

print(analyzer.report())

# Check acceptability
if results['indices']['Cpk'] < 1.33:
    print("WARNING: Process not capable - Cpk < 1.33")
```

### Gage R&R (Measurement System Analysis)

Validate measurement system before deploying models:

```python
from foodspec.qc.gage_rr import GageRR

# Crossed design: each operator measures each part
gage_rr = GageRR()

results = gage_rr.analyze_crossed(
    measurements=measurements_array,
    parts=part_ids,
    operators=operator_ids,
    tolerance=20.0
)

print(gage_rr.report())

# Check acceptability
if results['percent_tolerance']['gage_rr'] > 30:
    print("REJECT: Measurement system not acceptable")
elif results['ndc'] < 5:
    print("WARNING: NDC < 5, system cannot distinguish enough categories")
```

---

## 3. Uncertainty Quantification

### Prediction Intervals with Bootstrap

```python
from foodspec.trust.regression_uncertainty import BootstrapPredictionIntervals
from sklearn.ensemble import RandomForestRegressor

# Fit bootstrap ensemble
pi = BootstrapPredictionIntervals(n_bootstrap=100, confidence=0.95)
pi.fit(X_train, y_train, RandomForestRegressor, random_state=42)

# Generate predictions with intervals
result = pi.predict(X_test, method="bca")  # Bias-corrected and accelerated

predictions_with_uncertainty = pd.DataFrame({
    'y_pred': result['mean'],
    'y_lower': result['lower'],
    'y_upper': result['upper'],
    'uncertainty': result['std']
})
```

### Quantile Regression

```python
from foodspec.trust.regression_uncertainty import QuantileRegression
from sklearn.linear_model import Ridge

# Fit quantile regression models
qr = QuantileRegression(
    quantiles=[0.05, 0.5, 0.95],
    confidence=0.90
)
qr.fit(X_train, y_train, Ridge)

# Predictions at multiple quantiles
result = qr.predict(X_test)
# result['q0.05'] -> lower bound
# result['q0.5']  -> median
# result['q0.95'] -> upper bound
```

### Conformal Prediction

Distribution-free prediction intervals:

```python
from foodspec.trust.regression_uncertainty import ConformalRegression
from sklearn.linear_model import LinearRegression

# Train base model
model = LinearRegression()
model.fit(X_train, y_train)

# Wrap with conformal prediction
conformal = ConformalRegression(confidence=0.95, method="standard")
conformal.fit(X_train, y_train, model)

# Get guaranteed coverage
result = conformal.predict(X_test)
# Lower and upper bounds have at least 95% empirical coverage
```

---

## 4. Multivariate Drift Detection

### MMD Drift Detection

Maximum Mean Discrepancy for high-dimensional monitoring:

```python
from foodspec.qc.drift_multivariate import MMDDriftDetector

# Initialize with reference distribution
detector = MMDDriftDetector(kernel="rbf", alpha=0.05)
detector.initialize(X_reference)

# Monitor incoming data for drift
result = detector.detect(X_test)

if result['is_drift']:
    print(f"Drift detected! MMD²={result['mmd2']:.4f} > threshold={result['threshold']:.4f}")
```

### Wasserstein Drift

Optimal transport distance:

```python
from foodspec.qc.drift_multivariate import WassersteinDriftDetector

detector = WassersteinDriftDetector(
    approximation="sliced",  # Fast sliced Wasserstein
    n_projections=100
)
detector.initialize(X_reference)

result = detector.detect(X_test)
print(f"Wasserstein distance: {result['wasserstein_distance']:.4f}")
```

### Consensus Monitoring

Combine multiple drift detectors:

```python
from foodspec.qc.drift_multivariate import MultivariateDriftMonitor

monitor = MultivariateDriftMonitor()
monitor.initialize(X_reference)

# Monitor continuously
for batch in data_stream:
    result = monitor.detect(batch)
    if result['is_drift']:
        # Multiple methods agree on drift
        print(f"Consensus: {result['drift_votes']}/{result['n_methods']} detectors triggered")

# Get summary
summary = monitor.get_drift_summary()
print(f"Drift rate: {summary['drift_rate']:.1%}")
```

---

## 5. Decision Policy & Audit Logs

### Cost-Sensitive Decision Making

```python
from foodspec.trust.decision_policy import RegulatoryDecisionPolicy

# Create policy with cost matrix
policy = RegulatoryDecisionPolicy(
    cost_matrix={
        "cost_fp": 10.0,  # High cost for false accepts
        "cost_fn": 1.0,   # Lower cost for false rejects
        "cost_tp": 0.0,
        "cost_tn": 0.0,
    },
    utilities={
        "accept": 1.0,
        "reject": -0.5,
        "investigate": 0.1,
    }
)

# Make decision with full audit trail
result = policy.decide(
    scores={
        "accept": 0.8,
        "reject": 0.15,
        "investigate": 0.05
    },
    context={
        "sample_id": "SAMPLE_001",
        "batch_id": "BATCH_12345",
        "measurement_date": "2024-01-15",
    }
)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Expected utility: {result['expected_utility']:.4f}")

# Export audit log
audit_json = policy.get_audit_log()
with open("audit_trail.json", "w") as f:
    f.write(audit_json)
```

### Operating Point Calibration

```python
# Calibrate thresholds to achieve target false positive rate
calibration = policy.calibrate_costs(
    y_true=y_validation,
    y_pred_proba=pred_proba_validation,
    target_fp_rate=0.05  # Accept at most 5% false positives
)

print(f"Recommended threshold: {calibration['recommended_threshold']:.4f}")
print(f"Achieved FPR: {calibration['actual_fpr']:.2%}")
print(f"Achieved TPR: {calibration['actual_tpr']:.2%}")
```

---

## 6. Governance & Metadata

### Instrument Registration

```python
from foodspec.data.governance import InstrumentProfile, GovernanceRegistry

# Create instrument profile
profile = InstrumentProfile(
    instrument_id="FTIR_LAB_001",
    instrument_type="FTIR",
    manufacturer="PerkinElmer",
    model="Spectrum Two",
    serial_number="SN2024-12345",
    installation_date="2024-01-01",
    calibration_interval_days=365,
    location="Lab 202",
    responsible_person="J. Smith"
)

# Register with governance system
registry = GovernanceRegistry()
registry.register_instrument(profile)

print(f"Registered: {profile.get_fingerprint()}")
```

### Calibration Tracking

```python
from foodspec.data.governance import CalibrationRecord

# Log calibration event
calibration = CalibrationRecord(
    calibration_id="CAL_2024_001",
    instrument_id="FTIR_LAB_001",
    calibration_date="2024-01-15",
    calibration_method="Direct comparison to NIST SRM 1507",
    standard_reference="NIST SRM 1507",
    performed_by="J. Smith",
    acceptance_criteria={"rmse": 0.05, "r2": 0.995},
    results={"rmse": 0.038, "r2": 0.9973},
    passed=True,
    valid_until="2025-01-15",
    calibration_points=20,
)

registry.add_calibration(calibration)

# Print certificate
print(calibration.get_certificate())

# Verify calibration status
is_valid = registry.is_instrument_calibrated("FTIR_LAB_001")
```

### Environmental Conditions

```python
from foodspec.data.governance import EnvironmentLog

# Log environmental conditions during measurement
env_log = EnvironmentLog(
    log_id="ENV_2024_001",
    instrument_id="FTIR_LAB_001",
    timestamp="2024-01-15T10:30:00",
    temperature_c=22.5,
    humidity_percent=45.0,
    air_pressure_hpa=1013.2,
    notes="Stable conditions"
)

registry.add_environment_log(env_log)

# Validate measurement conditions
validation = registry.validate_measurement_conditions(
    instrument_id="FTIR_LAB_001",
    measurement_timestamp="2024-01-15T10:35:00",
    allowed_deviation_minutes=30
)

if not validation['valid']:
    print(f"REJECT: {validation['reason']}")
```

---

## 7. Regulatory Dossier

### Generate Regulatory Submission Package

```python
from foodspec.reporting.dossier import RegulatoryDossierGenerator

# Create dossier
dossier = RegulatoryDossierGenerator(
    model_id="SPECTRAL_CLASSIFICATION_V2",
    version="2.0.0"
)

# Add sections
dossier.add_model_card(
    model_type="Random Forest",
    intended_use="Spectral classification for food quality assessment",
    developers=["J. Doe", "M. Smith"],
    training_data={
        "samples": 5000,
        "features": 2048,
        "wavelength_range": "400-2500 nm",
    },
    limitations=[
        "Calibrated for type A samples only",
        "Not validated for temperatures outside 15-30°C",
    ]
)

dossier.add_validation_results(
    metrics={
        "accuracy": 0.952,
        "precision": 0.945,
        "recall": 0.958,
        "f1_score": 0.951,
    },
    test_set_size=1000,
    test_set_characteristics={
        "class_balance": "balanced",
        "representativeness": "includes rare categories",
    }
)

dossier.add_uncertainty_section(
    method="Bootstrap BCa with 1000 replications",
    calibration_results={
        "coverage_at_95": 0.951,
        "mean_interval_width": 0.045,
    },
    prediction_intervals={
        "method": "bias-corrected and accelerated",
        "coverage": 0.95,
    }
)

dossier.add_drift_monitoring(
    monitoring_frequency="Weekly",
    drift_detector_type="MMD + Wasserstein consensus",
    alert_thresholds={
        "mmd2": 0.05,
        "wasserstein": 0.1,
    },
    remediation_steps=[
        "Alert QA on drift detection",
        "Initiate re-calibration procedure",
        "Hold model in conservative mode pending validation",
        "Generate incident report",
    ]
)

dossier.add_decision_rules(
    policy_type="Cost-sensitive utility maximization",
    decision_rules={
        "accept": "p_accept >= 0.80",
        "reject": "p_reject >= 0.30 OR uncertainty > 0.05",
        "investigate": "0.30 <= p_accept < 0.80",
    },
    cost_matrix={
        "cost_fp": 10.0,  # High cost for false accepts
        "cost_fn": 1.0,
    },
    audit_requirements={
        "log_all_decisions": True,
        "retain_logs": "7 years",
    }
)

dossier.add_governance(
    responsible_personnel=[
        "Jane Doe (Model Owner)",
        "Bob Smith (QA Lead)",
        "Alice Johnson (Regulatory Affairs)"
    ],
    approval_date="2024-01-20",
    review_schedule="Annual",
    data_sources=[
        "Instrument: FTIR_LAB_001",
        "Environment: Controlled lab conditions",
        "Samples: Supplier approved reference materials"
    ],
    instrument_profiles={
        "FTIR_LAB_001": {
            "manufacturer": "PerkinElmer",
            "last_calibration": "2024-01-15",
        }
    }
)

# Add compliance checklist
checklist = RegulatoryDossierGenerator.standard_compliance_checklist()
# Mark items as complete
checklist.update({k: True for k in checklist.keys()})
dossier.add_compliance_checklist(checklist)

# Add audit trail
dossier.add_audit_trail(
    policy_decisions=[
        {
            "timestamp": "2024-01-20T10:30:00",
            "decision": "APPROVED",
            "by": "Regulatory Affairs",
        }
    ],
    key_events=[
        "Model trained: 2024-01-10",
        "Validation completed: 2024-01-15",
        "Dossier generated: 2024-01-20",
    ]
)

# Save in multiple formats
files = dossier.save(
    output_dir="/regulatory_submissions/",
    formats=["json", "markdown"]  # PDF requires reportlab
)

print(f"Dossier saved to: {files}")

# Version locking
fingerprint = dossier.lock_version()
print(f"Version lock fingerprint: {fingerprint}")
```

---

## 8. Compliance Checklist

### Standard Regulatory Items

```python
checklist = RegulatoryDossierGenerator.standard_compliance_checklist()

# Example checklist completion
compliance = {
    "Model documentation complete": True,
    "Intended use clearly defined": True,
    "Validation study performed and documented": True,
    "Uncertainty quantification implemented": True,
    "Drift monitoring plan established": True,
    "Decision policy documented and tested": True,
    "Governance metadata captured": True,
    "Audit trail enabled and tested": True,
    "Calibration verified with standards": True,
    "Environmental conditions documented": True,
    "Training data provenance documented": True,
    "Test data representative and balanced": True,
    "Model performance acceptable": True,
    "Risk assessment completed": True,
    "Approval by responsible person obtained": True,
    "Version lock and fingerprint verified": True,
}
```

---

## 9. Best Practices

### Before Production Deployment

1. **Complete Validation Study**
   - Minimum 3-independent test sets
   - Include edge cases and rare categories
   - Document performance for each category

2. **Uncertainty Validation**
   - Verify 95% coverage at nominal confidence level
   - Check interval width appropriateness
   - Compare methods (bootstrap vs. quantile vs. conformal)

3. **Drift Monitoring**
   - Establish baseline from reference data
   - Define alert thresholds based on business requirements
   - Test remediation procedures

4. **Decision Policy**
   - Calibrate cost matrix to business requirements
   - Validate operating point on independent data
   - Document threshold selection rationale

5. **Governance**
   - Register all instruments with calibration history
   - Document environmental conditions continuously
   - Maintain 7+ years audit trail

### During Production

1. **Continuous Monitoring**
   - Check drift detectors weekly
   - Review audit logs monthly
   - Validate prediction intervals on new data

2. **Incident Management**
   - Document any threshold exceedances
   - Initiate corrective actions per decision policy
   - Generate incident reports with full context

3. **Periodic Review**
   - Annual model performance review
   - Recalibrate decision policies as business needs evolve
   - Update dossier with new validation data

---

## 10. Regulatory Submission

### FDA/EMA Requirements

Regulatory agencies expect:

1. **Comprehensive Documentation**
   - Auto-generated dossier with all sections
   - Version-locked with SHA256 fingerprints
   - Complete audit trail of decisions

2. **Validation Evidence**
   - Uncertainty quantification with coverage proof
   - Drift monitoring plan with alert procedures
   - Performance on diverse datasets

3. **Governance Framework**
   - Instrument calibration certificates
   - Environmental condition logs
   - Personnel signatures and approvals

4. **Maintenance Plan**
   - Drift monitoring schedule
   - Recalibration procedures
   - Model update policy

### Submission Artifacts

```
regulatory_submissions/
├── MODEL_ID_regulatory_dossier_v2.0.0.json
├── MODEL_ID_regulatory_dossier_v2.0.0.md
├── MODEL_ID_regulatory_dossier_v2.0.0.pdf  (if reportlab available)
├── audit_trails/
│   └── policy_audit_log_2024.json
├── governance_registry.json
├── calibration_certificates/
│   ├── FTIR_LAB_001_2024-01-15.json
│   └── FTIR_LAB_001_2024-01-15.pdf
└── compliance_checklist.json
```

---

## 11. Example: Complete Regulatory Workflow

```python
# 1. Prepare data and train model
X_train, y_train = load_training_data()
X_val, y_val = load_validation_data()
X_test, y_test = load_test_data()

model = train_random_forest(X_train, y_train)

# 2. Quantify uncertainty
pi = BootstrapPredictionIntervals(n_bootstrap=100)
pi.fit(X_train, y_train, RandomForestRegressor)

# 3. Setup governance
registry = GovernanceRegistry()
registry.register_instrument(InstrumentProfile(...))
registry.add_calibration(CalibrationRecord(...))

# 4. Monitor drift
monitor = MultivariateDriftMonitor()
monitor.initialize(X_train)

# 5. Setup decision policy
policy = RegulatoryDecisionPolicy()

# 6. Generate dossier
dossier = RegulatoryDossierGenerator("MODEL_001", "1.0.0")
dossier.add_model_card(...)
dossier.add_validation_results(...)
dossier.add_uncertainty_section(...)
dossier.add_drift_monitoring(...)
dossier.add_decision_rules(...)
dossier.add_governance(...)
dossier.add_compliance_checklist(checklist)
dossier.add_audit_trail(...)

files = dossier.save("/regulatory_submission/", formats=["json", "markdown"])

# 7. Submit for approval
print(f"Regulatory dossier ready: {files}")
```

---

## References

- FDA Guidance on Machine Learning and Artificial Intelligence in Regulatory Decision-Making
- 21 CFR Part 11: Electronic Records; Electronic Signatures
- ICH Q14: Analytical Procedure Development
- JCGM 100: Guide to the Expression of Uncertainty in Measurement
