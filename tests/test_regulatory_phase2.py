"""Phase 2 Regulatory Platform Tests: QC, Uncertainty, Drift, Policy, Governance, Dossier."""

import numpy as np
import pytest

# Governance Tests
from foodspec.data.governance import (
    CalibrationRecord,
    EnvironmentLog,
    GovernanceRegistry,
    InstrumentProfile,
)

# QC Tests
from foodspec.qc.capability import CapabilityIndices, CUSUMChart, ProcessCapability
from foodspec.qc.drift_multivariate import MMDDriftDetector, MultivariateDriftMonitor, WassersteinDriftDetector
from foodspec.qc.gage_rr import GageRR

# Dossier Tests
from foodspec.reporting.dossier import RegulatoryDossierGenerator

# Decision Policy Tests
from foodspec.trust.decision_policy import (
    CostSensitiveROC,
    PolicyAuditLog,
    RegulatoryDecisionPolicy,
    UtilityMaximizer,
)

# Uncertainty Tests
from foodspec.trust.regression_uncertainty import (
    BootstrapPredictionIntervals,
    ConformalRegression,
    QuantileRegression,
)

# ============================================================================
# FIXTURE: Sample Data
# ============================================================================

@pytest.fixture
def reference_data():
    """Generate reference QC data."""
    np.random.seed(42)
    return np.random.normal(100, 5, 100)


@pytest.fixture
def process_measurements():
    """Generate process measurement data."""
    np.random.seed(42)
    return np.random.normal(100, 3, 50)


@pytest.fixture
def reference_multivariate():
    """Generate multivariate reference data."""
    np.random.seed(42)
    return np.random.normal(0, 1, (200, 5))


@pytest.fixture
def test_multivariate_drift():
    """Generate multivariate test data with drift."""
    np.random.seed(42)
    # Shifted distribution
    return np.random.normal(0.5, 1.2, (50, 5))


@pytest.fixture
def measurement_data():
    """Generate Gage R&R data (crossed design)."""
    np.random.seed(42)
    n_parts = 10
    n_operators = 3
    n_replicates = 2

    measurements = []
    parts = []
    operators = []

    for p in range(n_parts):
        for o in range(n_operators):
            for r in range(n_replicates):
                # True part effect + operator effect + measurement error
                meas = 100 + p * 2 + o * 0.5 + np.random.normal(0, 0.3)
                measurements.append(meas)
                parts.append(p + 1)
                operators.append(o + 1)

    return np.array(measurements), np.array(parts), np.array(operators)


# ============================================================================
# QC TESTS
# ============================================================================

class TestCUSUMChart:
    """Test CUSUM control chart functionality."""

    def test_initialization(self):
        """Test CUSUM chart creation."""
        chart = CUSUMChart(target=100, k=0.5, h=4.77)
        assert chart.target == 100
        assert chart.k == 0.5

    def test_initialize_from_reference(self, reference_data):
        """Test initialization with reference data."""
        chart = CUSUMChart()
        chart.initialize(reference_data)
        assert chart.mean_ is not None
        assert chart.std_ > 0

    def test_update_single_observation(self, reference_data):
        """Test CUSUM update with single observation."""
        chart = CUSUMChart()
        chart.initialize(reference_data)
        C_pos, C_neg, is_alarm = chart.update(100)
        assert isinstance(C_pos, (int, float))
        assert isinstance(C_neg, (int, float))
        # is_alarm can be numpy bool or Python bool
        assert isinstance(is_alarm, (bool, np.bool_))

    def test_process_stream(self, reference_data):
        """Test CUSUM processing stream of data."""
        chart = CUSUMChart(h=5)
        chart.initialize(reference_data[:50])
        C_pos, C_neg, alarms = chart.process(reference_data[50:])
        assert len(C_pos) == len(reference_data[50:])
        assert np.sum(alarms) >= 0

    def test_run_length_statistics(self, reference_data):
        """Test run length calculation."""
        chart = CUSUMChart()
        chart.initialize(reference_data[:50])
        chart.process(reference_data[50:])
        stats = chart.get_run_length()
        assert "n_observations" in stats
        assert "run_length" in stats


class TestCapabilityIndices:
    """Test process capability analysis."""

    def test_basic_capability_calculation(self, process_measurements):
        """Test basic capability index calculation."""
        indices = CapabilityIndices.calculate(
            process_measurements,
            lower_spec=90,
            upper_spec=110,
        )
        assert "Pp" in indices
        assert "Ppk" in indices
        assert indices["Ppk"] > 0

    def test_capability_with_subgroups(self, process_measurements):
        """Test Cp/Cpk with subgrouped data."""
        indices = CapabilityIndices.calculate(
            process_measurements,
            lower_spec=90,
            upper_spec=110,
            sample_size=5,
        )
        assert "Cp" in indices
        assert "Cpk" in indices

    def test_capability_classification(self):
        """Test capability index classification."""
        classifications = [
            (1.67, "Excellent"),
            (1.33, "Very Good"),
            (1.0, "Good"),
            (0.67, "Acceptable"),
            (0.5, "Unacceptable"),
        ]
        for cpk, expected in classifications:
            result = CapabilityIndices.classify_capability(cpk)
            assert result == expected

    def test_process_capability_analysis(self, process_measurements):
        """Test comprehensive capability analysis."""
        analyzer = ProcessCapability(lower_spec=90, upper_spec=110)
        results = analyzer.analyze(process_measurements)
        assert results["n_observations"] == len(process_measurements)
        assert "indices" in results
        assert "capability_classification" in results


class TestGageRR:
    """Test Gage R&R measurement system analysis."""

    def test_gage_rr_crossed_design(self, measurement_data):
        """Test Gage R&R for crossed design."""
        measurements, parts, operators = measurement_data
        gage_rr = GageRR()
        results = gage_rr.analyze_crossed(measurements, parts, operators, tolerance=20)

        assert "variance_components" in results
        assert "percent_tolerance" in results
        assert results["percent_tolerance"]["gage_rr"] >= 0

    def test_gage_rr_acceptability_classification(self, measurement_data):
        """Test Gage R&R acceptability."""
        measurements, parts, operators = measurement_data
        gage_rr = GageRR()
        results = gage_rr.analyze_crossed(measurements, parts, operators, tolerance=20)

        assert "acceptability" in results
        assert "gage_rr" in results["acceptability"]
        assert "ndc" in results["acceptability"]

    def test_ndc_calculation(self, measurement_data):
        """Test Number of Distinct Categories."""
        measurements, parts, operators = measurement_data
        gage_rr = GageRR()
        results = gage_rr.analyze_crossed(measurements, parts, operators)
        assert results["ndc"] >= 0

    def test_gage_rr_report_generation(self, measurement_data):
        """Test report generation."""
        measurements, parts, operators = measurement_data
        gage_rr = GageRR()
        gage_rr.analyze_crossed(measurements, parts, operators, tolerance=20)
        report = gage_rr.report()
        assert isinstance(report, str)
        assert "Gage R&R" in report or "analysis" in report.lower()


# ============================================================================
# UNCERTAINTY TESTS
# ============================================================================

class TestBootstrapPredictionIntervals:
    """Test bootstrap prediction interval estimation."""

    def test_bootstrap_fit(self):
        """Test bootstrap model fitting."""
        from sklearn.linear_model import LinearRegression

        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1

        pi = BootstrapPredictionIntervals(n_bootstrap=10, confidence=0.95)
        pi.fit(X, y, LinearRegression)

        assert len(pi.models_) == 10
        assert pi.confidence == 0.95

    def test_bootstrap_predictions_percentile(self):
        """Test percentile method predictions."""
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1
        X_test = np.random.randn(10, 3)

        pi = BootstrapPredictionIntervals(n_bootstrap=20)
        pi.fit(X, y, LinearRegression)
        result = pi.predict(X_test, method="percentile")

        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert len(result["mean"]) == len(X_test)
        assert np.all(result["lower"] <= result["mean"])
        assert np.all(result["mean"] <= result["upper"])

    def test_bootstrap_predictions_bca(self):
        """Test BCa method."""
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1

        pi = BootstrapPredictionIntervals(n_bootstrap=10)
        pi.fit(X, y, LinearRegression)
        result = pi.predict(X[:5], method="bca")

        assert "lower" in result
        assert "upper" in result


class TestQuantileRegression:
    """Test quantile regression."""

    def test_quantile_regression_fit(self):
        """Test quantile regression fitting."""
        from sklearn.linear_model import LinearRegression

        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50)

        qr = QuantileRegression(confidence=0.90)
        qr.fit(X, y, LinearRegression)

        assert len(qr.models_) == 3  # q=0.05, 0.5, 0.95

    def test_quantile_predictions(self):
        """Test quantile predictions."""
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.5

        qr = QuantileRegression(confidence=0.95)
        qr.fit(X, y, LinearRegression)
        result = qr.predict(X[:5])

        assert "median" in result
        assert "lower" in result
        assert "upper" in result


class TestConformalRegression:
    """Test conformal prediction."""

    def test_conformal_initialization(self):
        """Test conformal regressor creation."""
        from sklearn.linear_model import LinearRegression

        conf = ConformalRegression(confidence=0.95)
        model = LinearRegression()
        X = np.random.randn(30, 2)
        y = X[:, 0] + np.random.randn(30) * 0.1

        model.fit(X, y)
        conf.fit(X, y, model)

        assert conf.threshold_ is not None

    def test_conformal_predictions(self):
        """Test conformal interval predictions."""
        from sklearn.linear_model import LinearRegression

        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] * 2 + np.random.randn(50) * 0.1

        model = LinearRegression()
        model.fit(X, y)

        conf = ConformalRegression(confidence=0.95)
        conf.fit(X, y, model)

        result = conf.predict(X[:10])
        assert "lower" in result
        assert "upper" in result
        assert result["lower"] is not None
        assert result["upper"] is not None
        assert len(result["lower"]) == 10


# ============================================================================
# DRIFT DETECTION TESTS
# ============================================================================

class TestMMDDriftDetector:
    """Test MMD drift detection."""

    def test_mmd_initialization(self, reference_multivariate):
        """Test MMD detector initialization."""
        detector = MMDDriftDetector(kernel="rbf", alpha=0.05)
        detector.initialize(reference_multivariate)

        assert detector.reference_data_ is not None
        assert detector.mmd_threshold_ is not None

    def test_mmd_computation(self, reference_multivariate, test_multivariate_drift):
        """Test MMD distance computation."""
        detector = MMDDriftDetector()
        detector.initialize(reference_multivariate)

        mmd2 = detector.compute_mmd(test_multivariate_drift)
        assert isinstance(mmd2, float)
        assert mmd2 >= 0

    def test_mmd_drift_detection(self, reference_multivariate, test_multivariate_drift):
        """Test drift detection with MMD."""
        detector = MMDDriftDetector(alpha=0.1)
        detector.initialize(reference_multivariate)

        result = detector.detect(test_multivariate_drift)
        assert "is_drift" in result
        assert "mmd2" in result
        assert "threshold" in result


class TestWassersteinDriftDetector:
    """Test Wasserstein drift detection."""

    def test_wasserstein_initialization(self, reference_multivariate):
        """Test Wasserstein detector setup."""
        detector = WassersteinDriftDetector(approximation="sliced", n_projections=20)
        detector.initialize(reference_multivariate)

        assert detector.reference_data_ is not None

    def test_wasserstein_distance(self, reference_multivariate, test_multivariate_drift):
        """Test Wasserstein distance computation."""
        detector = WassersteinDriftDetector(approximation="sliced", n_projections=10)
        detector.initialize(reference_multivariate)

        distance = detector.compute_distance(test_multivariate_drift)
        assert isinstance(distance, float)
        assert distance >= 0

    def test_wasserstein_detection(self, reference_multivariate):
        """Test drift detection with Wasserstein."""
        detector = WassersteinDriftDetector(n_projections=20)
        detector.initialize(reference_multivariate)

        # Generate stable test data (no drift)
        test_stable = np.random.normal(0, 1, (50, 5))
        result = detector.detect(test_stable, threshold=0.1)

        assert "is_drift" in result
        assert "wasserstein_distance" in result


class TestMultivariateDriftMonitor:
    """Test integrated drift monitoring."""

    def test_monitor_initialization(self, reference_multivariate):
        """Test monitor initialization."""
        monitor = MultivariateDriftMonitor()
        monitor.initialize(reference_multivariate)

        assert monitor.mmd_detector is not None
        assert monitor.wasserstein_detector is not None

    def test_monitor_detection(self, reference_multivariate):
        """Test consensus drift detection."""
        monitor = MultivariateDriftMonitor()
        monitor.initialize(reference_multivariate)

        test_data = np.random.normal(0, 1, (30, 5))
        result = monitor.detect(test_data)

        assert "is_drift" in result
        assert "drift_votes" in result
        assert len(monitor.history_) == 1

    def test_monitor_summary(self, reference_multivariate):
        """Test monitoring summary statistics."""
        monitor = MultivariateDriftMonitor()
        monitor.initialize(reference_multivariate)

        for _ in range(3):
            test_data = np.random.normal(0, 1, (20, 5))
            monitor.detect(test_data)

        summary = monitor.get_drift_summary()
        assert summary["n_samples_monitored"] == 3


# ============================================================================
# DECISION POLICY TESTS
# ============================================================================

class TestPolicyAuditLog:
    """Test audit logging."""

    def test_audit_log_creation(self):
        """Test audit log initialization."""
        log = PolicyAuditLog("policy_001")
        assert log.policy_id == "policy_001"
        assert len(log.entries) == 0

    def test_log_decision(self):
        """Test decision logging."""
        log = PolicyAuditLog("policy_001")
        log.log_decision(
            decision="ACCEPT",
            context={"sample_id": "S123"},
            confidence=0.95,
            reasoning="Passed all QC checks",
            parameters={"threshold": 0.5},
        )
        assert len(log.entries) == 1
        assert log.entries[0]["decision"] == "ACCEPT"

    def test_audit_log_export(self):
        """Test JSON export."""
        log = PolicyAuditLog("policy_001")
        log.log_decision("ACCEPT", {}, 0.95, "Test", {})
        json_str = log.to_json()
        assert "policy_001" in json_str
        assert "ACCEPT" in json_str

    def test_audit_log_summary(self):
        """Test summary statistics."""
        log = PolicyAuditLog("policy_001")
        log.log_decision("ACCEPT", {}, 0.95, "Test", {})
        log.log_decision("REJECT", {}, 0.70, "Test", {})

        summary = log.summary()
        assert summary["n_entries"] == 2
        assert "decision_counts" in summary


class TestCostSensitiveROC:
    """Test cost-sensitive ROC analysis."""

    def test_cost_sensitive_analysis(self):
        """Test cost matrix optimization."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.6, 0.4])

        roc = CostSensitiveROC(cost_fp=2.0, cost_fn=1.0)
        result = roc.analyze(y_true, y_pred)

        assert "optimal_threshold" in result
        assert "optimal_cost" in result
        assert result["optimal_threshold"] >= 0
        assert result["optimal_threshold"] <= 1


class TestUtilityMaximizer:
    """Test utility maximization."""

    def test_decision_maximization(self):
        """Test utility-based decision making."""
        utilities = {"accept": 1.0, "reject": -0.5, "investigate": 0.2}
        maximizer = UtilityMaximizer(utilities)

        scores = {"accept": 0.7, "reject": 0.2, "investigate": 0.1}
        decision, eu, reasoning = maximizer.decide(scores)

        assert decision == "accept"
        assert eu > 0


class TestRegulatoryDecisionPolicy:
    """Test regulatory decision policy."""

    def test_policy_creation(self):
        """Test policy initialization."""
        policy = RegulatoryDecisionPolicy()
        assert policy.policy_id is not None
        assert policy.audit_log is not None

    def test_policy_decision(self):
        """Test decision making with audit."""
        policy = RegulatoryDecisionPolicy()

        scores = {"accept": 0.8, "reject": 0.1, "investigate": 0.1}
        result = policy.decide(scores, context={"sample_id": "S001"})

        assert "decision" in result
        assert "confidence" in result
        assert result["policy_id"] == policy.policy_id

    def test_audit_retrieval(self):
        """Test audit log retrieval."""
        policy = RegulatoryDecisionPolicy()
        policy.decide({"accept": 0.9, "reject": 0.05, "investigate": 0.05})

        audit_json = policy.get_audit_log()
        assert "entries" in audit_json

        summary = policy.get_audit_summary()
        assert summary["n_entries"] == 1


# ============================================================================
# GOVERNANCE TESTS
# ============================================================================

class TestInstrumentProfile:
    """Test instrument metadata."""

    def test_profile_creation(self):
        """Test instrument profile creation."""
        profile = InstrumentProfile(
            instrument_id="FTIR_001",
            instrument_type="FTIR",
            manufacturer="PerkinElmer",
            model="Spectrum Two",
            serial_number="SN12345",
            installation_date="2023-01-15",
        )
        assert profile.instrument_id == "FTIR_001"
        assert profile.calibration_status == "Valid"

    def test_profile_fingerprint(self):
        """Test profile integrity verification."""
        profile = InstrumentProfile(
            instrument_id="FTIR_001",
            instrument_type="FTIR",
            manufacturer="PerkinElmer",
            model="Spectrum Two",
            serial_number="SN12345",
            installation_date="2023-01-15",
        )
        fp1 = profile.get_fingerprint()
        fp2 = profile.get_fingerprint()
        assert fp1 == fp2  # Deterministic


class TestCalibrationRecord:
    """Test calibration history."""

    def test_calibration_record(self):
        """Test calibration record creation."""
        record = CalibrationRecord(
            calibration_id="CAL_001",
            instrument_id="FTIR_001",
            calibration_date="2024-01-15",
            calibration_method="Direct comparison to standard",
            standard_reference="NIST SRM 1507",
            performed_by="J. Smith",
            acceptance_criteria={"rmse": 0.05},
            results={"rmse": 0.04},
            passed=True,
        )
        assert record.passed is True

    def test_calibration_certificate(self):
        """Test certificate generation."""
        record = CalibrationRecord(
            calibration_id="CAL_001",
            instrument_id="FTIR_001",
            calibration_date="2024-01-15",
            calibration_method="Method A",
            standard_reference="Standard",
            performed_by="Smith",
            acceptance_criteria={"rmse": 0.05},
            results={"rmse": 0.04},
            passed=True,
        )
        cert = record.get_certificate()
        assert "CALIBRATION CERTIFICATE" in cert
        assert "PASSED" in cert


class TestGovernanceRegistry:
    """Test governance metadata registry."""

    def test_registry_initialization(self):
        """Test registry creation."""
        registry = GovernanceRegistry()
        assert len(registry.instruments) == 0
        assert len(registry.calibrations) == 0

    def test_register_instrument(self):
        """Test instrument registration."""
        registry = GovernanceRegistry()
        profile = InstrumentProfile(
            instrument_id="FTIR_001",
            instrument_type="FTIR",
            manufacturer="PerkinElmer",
            model="Spectrum Two",
            serial_number="SN12345",
            installation_date="2023-01-15",
        )
        registry.register_instrument(profile)
        assert "FTIR_001" in registry.instruments

    def test_add_calibration(self):
        """Test calibration record addition."""
        registry = GovernanceRegistry()
        profile = InstrumentProfile(
            instrument_id="FTIR_001",
            instrument_type="FTIR",
            manufacturer="PerkinElmer",
            model="Spectrum Two",
            serial_number="SN12345",
            installation_date="2023-01-15",
        )
        registry.register_instrument(profile)

        record = CalibrationRecord(
            calibration_id="CAL_001",
            instrument_id="FTIR_001",
            calibration_date="2024-01-15",
            calibration_method="Method A",
            standard_reference="Standard",
            performed_by="Smith",
            acceptance_criteria={"rmse": 0.05},
            results={"rmse": 0.04},
            passed=True,
        )
        registry.add_calibration(record)
        assert "CAL_001" in registry.calibrations

    def test_is_instrument_calibrated(self):
        """Test calibration status check."""
        registry = GovernanceRegistry()
        profile = InstrumentProfile(
            instrument_id="FTIR_001",
            instrument_type="FTIR",
            manufacturer="PerkinElmer",
            model="Spectrum Two",
            serial_number="SN12345",
            installation_date="2023-01-15",
        )
        registry.register_instrument(profile)
        assert registry.is_instrument_calibrated("FTIR_001") is True


class TestEnvironmentLog:
    """Test environmental condition logging."""

    def test_environment_log_creation(self):
        """Test environment log creation."""
        log = EnvironmentLog(
            log_id="ENV_001",
            instrument_id="FTIR_001",
            timestamp="2024-01-15T10:00:00",
            temperature_c=22.5,
            humidity_percent=45,
            air_pressure_hpa=1013,
        )
        assert log.temperature_c == 22.5

    def test_within_limits_check(self):
        """Test environmental limits validation."""
        log = EnvironmentLog(
            log_id="ENV_001",
            instrument_id="FTIR_001",
            timestamp="2024-01-15T10:00:00",
            temperature_c=22.5,
            humidity_percent=45,
            air_pressure_hpa=1013,
        )
        assert log.is_within_limits() is True

        log_bad = EnvironmentLog(
            log_id="ENV_002",
            instrument_id="FTIR_001",
            timestamp="2024-01-15T10:00:00",
            temperature_c=28.0,  # Outside default range
            humidity_percent=45,
            air_pressure_hpa=1013,
        )
        assert log_bad.is_within_limits() is False


# ============================================================================
# DOSSIER TESTS
# ============================================================================

class TestRegulatoryDossierGenerator:
    """Test regulatory dossier generation."""

    def test_dossier_initialization(self):
        """Test dossier creation."""
        dossier = RegulatoryDossierGenerator("MODEL_001", version="1.0.0")
        assert dossier.model_id == "MODEL_001"
        assert dossier.version == "1.0.0"

    def test_add_model_card(self):
        """Test model card section."""
        dossier = RegulatoryDossierGenerator("MODEL_001")
        dossier.add_model_card(
            model_type="Random Forest",
            intended_use="Spectral analysis",
            developers=["J. Doe"],
            training_data={"samples": 1000},
            limitations=["Works for type A samples only"],
        )
        assert any("model_card" in key.lower() for key in dossier.sections.keys())

    def test_fingerprint_generation(self):
        """Test version locking with fingerprints."""
        dossier = RegulatoryDossierGenerator("MODEL_001")
        dossier.add_model_card(
            model_type="RF", intended_use="Test", developers=[], training_data={}, limitations=[]
        )
        fp1 = dossier.get_fingerprint()
        fp2 = dossier.get_fingerprint()
        assert fp1 == fp2  # Deterministic
        assert len(fp1) == 64  # SHA256

    def test_version_locking(self):
        """Test version lock mechanism."""
        dossier = RegulatoryDossierGenerator("MODEL_001")
        fp = dossier.lock_version()
        assert len(fp) == 64
        assert len(dossier.fingerprint_chain) == 1

    def test_dossier_export_json(self):
        """Test JSON export."""
        dossier = RegulatoryDossierGenerator("MODEL_001")
        dossier.add_model_card("RF", "Test", [], {}, [])
        json_str = dossier.to_json()
        assert "MODEL_001" in json_str
        assert "1.0.0" in json_str

    def test_compliance_checklist(self):
        """Test compliance checklist."""
        checklist = RegulatoryDossierGenerator.standard_compliance_checklist()
        assert "Model documentation complete" in checklist
        assert len(checklist) == 16

    def test_dossier_add_all_sections(self):
        """Test comprehensive dossier with all sections."""
        dossier = RegulatoryDossierGenerator("MODEL_FULL", version="2.0.0")

        dossier.add_model_card(
            "Random Forest",
            "Spectral classification",
            ["J. Doe", "M. Smith"],
            {"samples": 1000, "features": 500},
            ["Limited to 400-2000 nm range"],
        )

        dossier.add_validation_data(
            {"accuracy": 0.95, "precision": 0.93},
            {"n_samples": 200, "class_balance": "balanced"},
            {"cross_validation": "5-fold"},
        )

        dossier.add_uncertainty_section(
            "Bootstrap",
            {"coverage": 0.95},
            {"mean_width": 0.1},
        )

        dossier.add_drift_monitoring(
            "Monthly",
            "MMD",
            {"threshold": 0.05},
            ["Recalibrate if drift detected"],
        )

        dossier.add_decision_rules(
            "Utility maximization",
            {"accept": 0.8, "reject": 0.2},
            {"cost_fp": 1.0, "cost_fn": 1.0},
            {"log_all_decisions": True},
        )

        dossier.add_governance(
            ["J. Doe (QA)"],
            "2024-01-15",
            "Annual",
            ["Training data warehouse"],
            {"FTIR_001": "PerkinElmer Spectrum Two"},
        )

        dossier.add_compliance_checklist(
            {k: True for k in RegulatoryDossierGenerator.standard_compliance_checklist().keys()}
        )

        dossier.add_audit_trail(
            ["Decision 1", "Decision 2"],
            ["Model created", "Model validated"],
        )

        # Verify all sections are present
        assert len(dossier.sections) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
