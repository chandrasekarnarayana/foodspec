import pytest

from foodspec.reporting.modes import (
    ModeConfig,
    ReportMode,
    get_mode_config,
    list_modes,
    validate_artifacts,
)


class TestReportModeEnum:
    def test_report_mode_values(self):
        """Test that all expected modes are defined."""
        assert ReportMode.RESEARCH.value == "research"
        assert ReportMode.REGULATORY.value == "regulatory"
        assert ReportMode.MONITORING.value == "monitoring"

    def test_report_mode_count(self):
        """Test that there are exactly three modes."""
        assert len(list(ReportMode)) == 3


class TestGetModeConfig:
    def test_get_mode_config_research(self):
        """Test RESEARCH mode configuration."""
        config = get_mode_config(ReportMode.RESEARCH)
        assert isinstance(config, ModeConfig)
        assert config.mode == ReportMode.RESEARCH
        assert "summary" in config.enabled_sections
        assert "metrics" in config.enabled_sections
        assert "uncertainty" in config.enabled_sections
        assert "manifest" in config.required_artifacts
        assert "metrics" in config.required_artifacts
        assert config.strictness_level == 1
        assert config.warnings_as_errors is False

    def test_get_mode_config_regulatory(self):
        """Test REGULATORY mode configuration."""
        config = get_mode_config(ReportMode.REGULATORY)
        assert isinstance(config, ModeConfig)
        assert config.mode == ReportMode.REGULATORY
        assert "qc" in config.enabled_sections
        assert "metrics" in config.enabled_sections
        assert "uncertainty" in config.enabled_sections
        assert "manifest" in config.required_artifacts
        assert "metrics" in config.required_artifacts
        assert "qc" in config.required_artifacts
        assert "protocol_snapshot" in config.required_artifacts
        assert "data_fingerprint" in config.required_artifacts
        assert config.strictness_level == 3
        assert config.warnings_as_errors is True

    def test_get_mode_config_monitoring(self):
        """Test MONITORING mode configuration."""
        config = get_mode_config(ReportMode.MONITORING)
        assert isinstance(config, ModeConfig)
        assert config.mode == ReportMode.MONITORING
        assert "summary" in config.enabled_sections
        assert "dataset" in config.enabled_sections
        assert "metrics" in config.enabled_sections
        assert "manifest" in config.required_artifacts
        assert "metrics" in config.required_artifacts
        assert config.strictness_level == 2
        assert config.warnings_as_errors is False

    def test_get_mode_config_by_string(self):
        """Test that modes can be retrieved by string (case-insensitive)."""
        config1 = get_mode_config("research")
        config2 = get_mode_config("RESEARCH")
        config3 = get_mode_config(ReportMode.RESEARCH)
        assert config1.mode == config2.mode == config3.mode == ReportMode.RESEARCH

    def test_get_mode_config_invalid_mode(self):
        """Test that invalid modes raise ValueError."""
        with pytest.raises(ValueError, match="Unknown reporting mode"):
            get_mode_config("invalid_mode")

    def test_get_mode_config_returns_consistent_instances(self):
        """Test that calling get_mode_config multiple times returns consistent data."""
        config1 = get_mode_config(ReportMode.RESEARCH)
        config2 = get_mode_config(ReportMode.RESEARCH)
        assert config1.enabled_sections == config2.enabled_sections
        assert config1.required_artifacts == config2.required_artifacts


class TestModeConfigAttributes:
    def test_all_modes_have_description(self):
        """Test that all modes have non-empty descriptions."""
        for mode in ReportMode:
            config = get_mode_config(mode)
            assert config.description
            assert len(config.description) > 5

    def test_all_modes_have_enabled_sections(self):
        """Test that all modes have at least some enabled sections."""
        for mode in ReportMode:
            config = get_mode_config(mode)
            assert len(config.enabled_sections) > 0

    def test_all_modes_have_required_artifacts(self):
        """Test that all modes define required artifacts."""
        for mode in ReportMode:
            config = get_mode_config(mode)
            assert len(config.required_artifacts) > 0
            assert "manifest" in config.required_artifacts
            assert "metrics" in config.required_artifacts

    def test_all_modes_have_default_plots(self):
        """Test that all modes define default plots."""
        for mode in ReportMode:
            config = get_mode_config(mode)
            assert len(config.default_plots) > 0

    def test_strictness_levels_vary(self):
        """Test that strictness levels differ across modes."""
        research_level = get_mode_config(ReportMode.RESEARCH).strictness_level
        regulatory_level = get_mode_config(ReportMode.REGULATORY).strictness_level
        monitoring_level = get_mode_config(ReportMode.MONITORING).strictness_level
        assert research_level < monitoring_level < regulatory_level

    def test_regulatory_has_strictest_checks(self):
        """Test that REGULATORY mode has strictest checks."""
        config = get_mode_config(ReportMode.REGULATORY)
        assert config.warnings_as_errors is True
        assert config.strictness_level == 3
        # Should require more artifacts
        other_reqs = max(
            len(get_mode_config(ReportMode.RESEARCH).required_artifacts),
            len(get_mode_config(ReportMode.MONITORING).required_artifacts),
        )
        assert len(config.required_artifacts) >= other_reqs

    def test_research_emphasizes_interpretation(self):
        """Test that RESEARCH mode emphasizes interpretation via plots."""
        config = get_mode_config(ReportMode.RESEARCH)
        # RESEARCH mode emphasizes analysis through feature importance, ROC, etc.
        assert "feature_importance" in config.default_plots
        assert "roc_curve" in config.default_plots

    def test_regulatory_emphasizes_qc(self):
        """Test that REGULATORY mode emphasizes QC."""
        config = get_mode_config(ReportMode.REGULATORY)
        assert "qc" in config.enabled_sections
        assert "qc" in config.required_artifacts
        assert "qc_pass_rate" in config.default_plots

    def test_monitoring_emphasizes_drift(self):
        """Test that MONITORING mode emphasizes drift and comparison."""
        config = get_mode_config(ReportMode.MONITORING)
        # MONITORING mode emphasizes drift and trend detection via plots
        assert "batch_drift" in config.default_plots
        assert "metric_trend" in config.default_plots


class TestListModes:
    def test_list_modes_returns_dict(self):
        """Test that list_modes returns a mapping."""
        modes = list_modes()
        assert isinstance(modes, dict)

    def test_list_modes_has_all_modes(self):
        """Test that list_modes includes all defined modes."""
        modes = list_modes()
        assert "research" in modes
        assert "regulatory" in modes
        assert "monitoring" in modes

    def test_list_modes_has_descriptions(self):
        """Test that all listed modes have descriptions."""
        modes = list_modes()
        for mode_name, description in modes.items():
            assert description
            assert len(description) > 5


class TestValidateArtifacts:
    def test_validate_all_artifacts_present_research(self):
        """Test validation passes when all artifacts are present."""
        config = get_mode_config(ReportMode.RESEARCH)
        available = config.required_artifacts + ["extra_artifact"]
        valid, missing = validate_artifacts(ReportMode.RESEARCH, available)
        assert valid is True
        assert len(missing) == 0

    def test_validate_missing_artifacts_research(self):
        """Test validation handles missing artifacts for non-strict mode."""
        valid, missing = validate_artifacts(
            ReportMode.RESEARCH,
            ["manifest"],
            warnings_as_errors=False,
        )
        assert valid is True  # Still valid because warnings_as_errors is False
        assert "metrics" in missing

    def test_validate_missing_artifacts_regulatory_strict(self):
        """Test that REGULATORY mode fails with missing artifacts."""
        config = get_mode_config(ReportMode.REGULATORY)
        valid, missing = validate_artifacts(
            ReportMode.REGULATORY,
            ["manifest"],
            warnings_as_errors=config.warnings_as_errors,
        )
        assert valid is False
        assert len(missing) > 0

    def test_validate_artifacts_by_string_mode(self):
        """Test that validate_artifacts works with string modes."""
        available = get_mode_config("research").required_artifacts
        valid, missing = validate_artifacts("research", available)
        assert valid is True
        assert len(missing) == 0

    def test_validate_artifacts_returns_missing_list(self):
        """Test that validation returns accurate missing list."""
        available = ["manifest"]
        valid, missing = validate_artifacts(ReportMode.REGULATORY, available)
        config = get_mode_config(ReportMode.REGULATORY)
        expected_missing = [a for a in config.required_artifacts if a not in available]
        assert set(missing) == set(expected_missing)

    def test_validate_artifacts_invalid_mode(self):
        """Test that validation raises error for invalid mode."""
        with pytest.raises(ValueError):
            validate_artifacts("invalid", ["manifest"])


class TestModeConsistency:
    def test_sections_are_unique_within_mode(self):
        """Test that sections are not duplicated within a mode."""
        for mode in ReportMode:
            config = get_mode_config(mode)
            sections = config.enabled_sections
            assert len(sections) == len(set(sections)), f"Duplicate sections in {mode.value}"

    def test_plots_are_unique_within_mode(self):
        """Test that plots are not duplicated within a mode."""
        for mode in ReportMode:
            config = get_mode_config(mode)
            plots = config.default_plots
            assert len(plots) == len(set(plots)), f"Duplicate plots in {mode.value}"

    def test_artifacts_are_unique_within_mode(self):
        """Test that required artifacts are not duplicated within a mode."""
        for mode in ReportMode:
            config = get_mode_config(mode)
            artifacts = config.required_artifacts
            assert len(artifacts) == len(set(artifacts)), f"Duplicate artifacts in {mode.value}"

    def test_strictness_is_positive(self):
        """Test that all modes have positive strictness levels."""
        for mode in ReportMode:
            config = get_mode_config(mode)
            assert config.strictness_level > 0
            assert config.strictness_level <= 3

    def test_mode_config_is_immutable(self):
        """Test that returned config is a dataclass (effectively immutable if treated correctly)."""
        config = get_mode_config(ReportMode.RESEARCH)
        assert isinstance(config, ModeConfig)
        # Attempting to mutate should not affect next call
        original_sections = config.enabled_sections.copy()
        config_again = get_mode_config(ReportMode.RESEARCH)
        assert config_again.enabled_sections == original_sections
