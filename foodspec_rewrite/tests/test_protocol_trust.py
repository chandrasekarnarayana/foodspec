"""
Tests for Trust protocol schema validation (Phase 9).

Verifies:
- All trust spec components validate correctly
- Invalid configurations raise actionable errors
- Defaults are explicit and sensible
- Integration with ProtocolV2 validation
"""

import pytest

from foodspec.core.protocol import (
    AbstentionRuleSpec,
    AbstentionSpec,
    CalibrationSpec,
    ConformalSpec,
    ProtocolV2,
    TrustInterpretabilitySpec,
    TrustSpec,
)


class TestCalibrationSpec:
    """Tests for CalibrationSpec."""

    def test_valid_methods(self) -> None:
        """Test all valid calibration methods."""
        for method in ["none", "platt", "isotonic"]:
            spec = CalibrationSpec(method=method)
            spec.validate_method()  # Should not raise

    def test_default_method(self) -> None:
        """Test default calibration method is 'none'."""
        spec = CalibrationSpec()
        assert spec.method == "none"

    def test_invalid_method(self) -> None:
        """Test invalid calibration method raises error."""
        spec = CalibrationSpec(method="bayesian")
        with pytest.raises(ValueError, match="Invalid calibration method"):
            spec.validate_method()

    def test_method_error_message(self) -> None:
        """Test error message is actionable."""
        spec = CalibrationSpec(method="invalid")
        with pytest.raises(ValueError) as exc_info:
            spec.validate_method()
        error_msg = str(exc_info.value)
        assert "invalid" in error_msg
        assert "none" in error_msg or "platt" in error_msg


class TestConformalSpec:
    """Tests for ConformalSpec."""

    def test_defaults(self) -> None:
        """Test default conformal configuration."""
        spec = ConformalSpec()
        assert spec.enabled is False
        assert spec.alpha == 0.1
        assert spec.condition_key is None

    def test_enabled_with_alpha(self) -> None:
        """Test enabling conformal with custom alpha."""
        spec = ConformalSpec(enabled=True, alpha=0.05)
        assert spec.enabled is True
        assert spec.alpha == 0.05

    def test_alpha_boundaries(self) -> None:
        """Test alpha value validation (0 < alpha <= 1)."""
        # Valid alphas
        ConformalSpec(alpha=0.01)
        ConformalSpec(alpha=0.5)
        ConformalSpec(alpha=1.0)

        # Alpha constraints are tested through Pydantic
        # We verify valid range works
        spec_small = ConformalSpec(alpha=0.001)
        assert spec_small.alpha == 0.001

    def test_condition_key_optional(self) -> None:
        """Test condition_key is optional."""
        spec1 = ConformalSpec()
        assert spec1.condition_key is None

        spec2 = ConformalSpec(condition_key="batch")
        assert spec2.condition_key == "batch"


class TestAbstentionRuleSpec:
    """Tests for AbstentionRuleSpec."""

    def test_max_prob_rule_valid(self) -> None:
        """Test valid max_prob abstention rule."""
        rule = AbstentionRuleSpec(type="max_prob", threshold=0.7)
        rule.validate_rule()  # Should not raise

    def test_conformal_size_rule_valid(self) -> None:
        """Test valid conformal_size abstention rule."""
        rule = AbstentionRuleSpec(type="conformal_size", max_size=3)
        rule.validate_rule()  # Should not raise

    def test_max_prob_missing_threshold(self) -> None:
        """Test max_prob rule requires threshold."""
        rule = AbstentionRuleSpec(type="max_prob")
        with pytest.raises(ValueError, match="threshold"):
            rule.validate_rule()

    def test_conformal_size_missing_max_size(self) -> None:
        """Test conformal_size rule requires max_size."""
        rule = AbstentionRuleSpec(type="conformal_size")
        with pytest.raises(ValueError, match="max_size"):
            rule.validate_rule()

    def test_invalid_rule_type(self) -> None:
        """Test invalid rule type raises error."""
        rule = AbstentionRuleSpec(type="entropy_threshold")
        with pytest.raises(ValueError, match="Invalid abstention rule type"):
            rule.validate_rule()

    def test_threshold_boundaries(self) -> None:
        """Test threshold is in (0, 1]."""
        from pydantic import ValidationError

        # Valid
        AbstentionRuleSpec(type="max_prob", threshold=0.01)
        AbstentionRuleSpec(type="max_prob", threshold=0.5)
        AbstentionRuleSpec(type="max_prob", threshold=1.0)

        # Invalid
        with pytest.raises(ValidationError):
            AbstentionRuleSpec(type="max_prob", threshold=0.0)
        with pytest.raises(ValidationError):
            AbstentionRuleSpec(type="max_prob", threshold=1.5)

    def test_max_size_boundaries(self) -> None:
        """Test max_size is >= 1."""
        from pydantic import ValidationError

        # Valid
        AbstentionRuleSpec(type="conformal_size", max_size=1)
        AbstentionRuleSpec(type="conformal_size", max_size=5)

        # Invalid
        with pytest.raises(ValidationError):
            AbstentionRuleSpec(type="conformal_size", max_size=0)


class TestAbstentionSpec:
    """Tests for AbstentionSpec."""

    def test_defaults(self) -> None:
        """Test default abstention configuration."""
        spec = AbstentionSpec()
        assert spec.enabled is False
        assert spec.rules == []
        assert spec.mode == "any"

    def test_enabled_with_rules(self) -> None:
        """Test enabling abstention with rules."""
        rules = [
            AbstentionRuleSpec(type="max_prob", threshold=0.7),
            AbstentionRuleSpec(type="conformal_size", max_size=3),
        ]
        spec = AbstentionSpec(enabled=True, rules=rules)
        assert spec.enabled is True
        assert len(spec.rules) == 2

    def test_valid_modes(self) -> None:
        """Test valid combination modes."""
        AbstentionSpec(mode="any")
        AbstentionSpec(mode="all")

    def test_invalid_mode(self) -> None:
        """Test invalid mode raises error."""
        spec = AbstentionSpec(mode="or")  # Should be "any"
        with pytest.raises(ValueError, match="Invalid abstention mode"):
            spec.validate_mode()

    def test_full_validation(self) -> None:
        """Test full abstention validation including rules."""
        rules = [AbstentionRuleSpec(type="max_prob", threshold=0.8)]
        spec = AbstentionSpec(enabled=True, rules=rules, mode="any")
        
        # Validate should not raise
        for rule in spec.rules:
            rule.validate_rule()
        spec.validate_mode()


class TestTrustInterpretabilitySpec:
    """Tests for TrustInterpretabilitySpec."""

    def test_defaults(self) -> None:
        """Test default interpretability configuration."""
        spec = TrustInterpretabilitySpec()
        assert spec.enabled is False
        assert spec.methods == []

    def test_valid_methods(self) -> None:
        """Test all valid interpretability methods."""
        valid_methods = ["coefficients", "permutation_importance", "marker_panels"]
        for method in valid_methods:
            spec = TrustInterpretabilitySpec(enabled=True, methods=[method])
            spec.validate_methods()  # Should not raise

    def test_multiple_methods(self) -> None:
        """Test multiple interpretability methods."""
        spec = TrustInterpretabilitySpec(
            enabled=True,
            methods=["coefficients", "permutation_importance"],
        )
        spec.validate_methods()  # Should not raise

    def test_invalid_method(self) -> None:
        """Test invalid method raises error."""
        spec = TrustInterpretabilitySpec(
            enabled=True,
            methods=["shapley_values"],
        )
        with pytest.raises(ValueError, match="Invalid interpretability method"):
            spec.validate_methods()

    def test_method_error_message(self) -> None:
        """Test error message is actionable."""
        spec = TrustInterpretabilitySpec(methods=["invalid_method"])
        with pytest.raises(ValueError) as exc_info:
            spec.validate_methods()
        error_msg = str(exc_info.value)
        assert "invalid_method" in error_msg
        assert "coefficients" in error_msg


class TestTrustSpec:
    """Tests for TrustSpec."""

    def test_defaults(self) -> None:
        """Test default trust configuration."""
        spec = TrustSpec()
        assert spec.calibration.method == "none"
        assert spec.conformal.enabled is False
        assert spec.abstention.enabled is False
        assert spec.interpretability.enabled is False

    def test_full_trust_config(self) -> None:
        """Test complete trust configuration."""
        spec = TrustSpec(
            calibration=CalibrationSpec(method="platt"),
            conformal=ConformalSpec(enabled=True, alpha=0.05),
            abstention=AbstentionSpec(
                enabled=True,
                rules=[AbstentionRuleSpec(type="max_prob", threshold=0.8)],
            ),
            interpretability=TrustInterpretabilitySpec(
                enabled=True,
                methods=["coefficients", "permutation_importance"],
            ),
        )
        spec.validate()  # Should not raise

    def test_validate_catches_invalid_calibration(self) -> None:
        """Test validate catches invalid calibration."""
        spec = TrustSpec(
            calibration=CalibrationSpec(method="bayesian"),
        )
        with pytest.raises(ValueError, match="calibration method"):
            spec.validate()

    def test_validate_catches_invalid_abstention_rule(self) -> None:
        """Test validate catches invalid abstention rule."""
        spec = TrustSpec(
            abstention=AbstentionSpec(
                enabled=True,
                rules=[AbstentionRuleSpec(type="max_prob")],  # Missing threshold
            ),
        )
        with pytest.raises(ValueError, match="threshold"):
            spec.validate()

    def test_validate_catches_invalid_mode(self) -> None:
        """Test validate catches invalid mode."""
        spec = TrustSpec(
            abstention=AbstentionSpec(mode="either"),  # Invalid
        )
        with pytest.raises(ValueError, match="mode"):
            spec.validate()

    def test_validate_catches_invalid_method(self) -> None:
        """Test validate catches invalid interpretability method."""
        spec = TrustSpec(
            interpretability=TrustInterpretabilitySpec(
                methods=["shapley"],
            ),
        )
        with pytest.raises(ValueError, match="interpretability method"):
            spec.validate()


class TestTrustProtocolV2Integration:
    """Tests for trust integration with ProtocolV2."""

    def test_protocol_with_trust_defaults(self) -> None:
        """Test ProtocolV2 has trust field with defaults."""
        protocol = ProtocolV2(
            version="2.0.0",
            data={"input": "data.csv", "modality": "raman", "label": "label"},
            task={"name": "classification", "objective": "classify"},
        )

        assert hasattr(protocol, "trust")
        assert protocol.trust.calibration.method == "none"
        assert protocol.trust.conformal.enabled is False
        assert protocol.trust.abstention.enabled is False
        assert protocol.trust.interpretability.enabled is False

    def test_protocol_with_custom_trust(self) -> None:
        """Test ProtocolV2 with custom trust configuration."""
        protocol = ProtocolV2(
            version="2.0.0",
            data={"input": "data.csv", "modality": "raman", "label": "label"},
            task={"name": "classification", "objective": "classify"},
            trust={
                "calibration": {"method": "isotonic"},
                "conformal": {"enabled": True, "alpha": 0.1},
                "abstention": {
                    "enabled": True,
                    "rules": [{"type": "max_prob", "threshold": 0.8}],
                },
                "interpretability": {
                    "enabled": True,
                    "methods": ["coefficients"],
                },
            },
        )

        assert protocol.trust.calibration.method == "isotonic"
        assert protocol.trust.conformal.enabled is True
        assert protocol.trust.abstention.enabled is True
        assert protocol.trust.interpretability.enabled is True
        assert "coefficients" in protocol.trust.interpretability.methods

    def test_protocol_validate_calls_trust_validate(self) -> None:
        """Test ProtocolV2.validate() calls trust.validate()."""
        protocol = ProtocolV2(
            version="2.0.0",
            data={
                "input": "data.csv",
                "modality": "raman",
                "label": "label",
                "metadata_map": {"sample_id": "sample_id", "modality": "modality", "label": "label"},
            },
            task={"name": "classification", "objective": "classify"},
            trust={
                "calibration": {"method": "invalid_method"},
            },
        )

        with pytest.raises(ValueError, match="calibration method"):
            protocol.validate()

    def test_protocol_validate_invalid_conformal_alpha(self) -> None:
        """Test ProtocolV2.validate() catches invalid alpha."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProtocolV2(
                version="2.0.0",
                data={"input": "data.csv", "modality": "raman", "label": "label"},
                task={"name": "classification", "objective": "classify"},
                trust={
                    "conformal": {"alpha": 1.5},  # Invalid: > 1
                },
            )

    def test_protocol_validate_invalid_abstention_rule(self) -> None:
        """Test ProtocolV2.validate() catches invalid abstention rule."""
        protocol = ProtocolV2(
            version="2.0.0",
            data={
                "input": "data.csv",
                "modality": "raman",
                "label": "label",
                "metadata_map": {"sample_id": "sample_id", "modality": "modality", "label": "label"},
            },
            task={"name": "classification", "objective": "classify"},
            trust={
                "abstention": {
                    "enabled": True,
                    "rules": [{"type": "max_prob"}],  # Missing threshold
                },
            },
        )

        with pytest.raises(ValueError, match="threshold"):
            protocol.validate()

    def test_protocol_validate_invalid_interpretability_method(self) -> None:
        """Test ProtocolV2.validate() catches invalid method."""
        protocol = ProtocolV2(
            version="2.0.0",
            data={
                "input": "data.csv",
                "modality": "raman",
                "label": "label",
                "metadata_map": {"sample_id": "sample_id", "modality": "modality", "label": "label"},
            },
            task={"name": "classification", "objective": "classify"},
            trust={
                "interpretability": {
                    "methods": ["invalid_method"],
                },
            },
        )

        with pytest.raises(ValueError, match="interpretability method"):
            protocol.validate()


class TestTrustProtocolSerialization:
    """Tests for trust config serialization in protocols."""

    def test_protocol_model_dump_includes_trust(self) -> None:
        """Test trust config is included in protocol dump."""
        protocol = ProtocolV2(
            version="2.0.0",
            data={"input": "data.csv", "modality": "raman", "label": "label"},
            task={"name": "classification", "objective": "classify"},
            trust={
                "calibration": {"method": "platt"},
            },
        )

        dumped = protocol.model_dump()
        assert "trust" in dumped
        assert dumped["trust"]["calibration"]["method"] == "platt"

    def test_protocol_from_dict_preserves_trust(self) -> None:
        """Test loading protocol from dict preserves trust config."""
        data = {
            "version": "2.0.0",
            "data": {"input": "data.csv", "modality": "raman", "label": "label"},
            "task": {"name": "classification", "objective": "classify"},
            "trust": {
                "calibration": {"method": "isotonic"},
                "conformal": {"enabled": True, "alpha": 0.05},
            },
        }

        protocol = ProtocolV2(**data)
        assert protocol.trust.calibration.method == "isotonic"
        assert protocol.trust.conformal.enabled is True
        assert protocol.trust.conformal.alpha == 0.05


__all__ = [
    "TestCalibrationSpec",
    "TestConformalSpec",
    "TestAbstentionRuleSpec",
    "TestAbstentionSpec",
    "TestTrustInterpretabilitySpec",
    "TestTrustSpec",
    "TestTrustProtocolV2Integration",
    "TestTrustProtocolSerialization",
]
