"""
Unit tests for regulatory readiness helpers.
"""

import pandas as pd
import pytest

from foodspec.trust.regulatory import (
    generate_trust_summary,
    integrity_checks,
    validate_reproducibility,
)


class TestValidateReproducibility:
    def test_valid_manifest(self):
        manifest = {
            "protocol_hash": "abc123",
            "seed": 42,
            "dependencies": {"numpy": "1.26.4", "pandas": "2.2.1"},
            "data_fingerprint": "sha256:deadbeef",
        }
        validate_reproducibility(manifest)

    @pytest.mark.parametrize(
        "manifest, error_msg",
        [
            ({}, "protocol_hash missing"),
            ({"protocol_hash": "abc"}, "seed missing"),
            ({"protocol_hash": "abc", "seed": 0}, "dependency versions"),
            ({"protocol_hash": "abc", "seed": 0, "dependencies": {}}, "dependency versions"),
            ({"protocol_hash": "abc", "seed": 0, "dependencies": {"numpy": "1.0"}}, "data_fingerprint"),
        ],
    )
    def test_invalid_manifest(self, manifest, error_msg):
        with pytest.raises(ValueError) as exc:
            validate_reproducibility(manifest)
        assert error_msg in str(exc.value)


class TestIntegrityChecks:
    def test_passes_on_valid_probabilities_and_labels(self):
        df = pd.DataFrame(
            {
                "prob_a": [0.7, 0.2],
                "prob_b": [0.3, 0.8],
                "predicted_class": [0, 1],
                "predicted_label": ["a", "b"],
            }
        )
        integrity_checks(df, prob_prefix="prob_")

    def test_fails_on_nan(self):
        df = pd.DataFrame(
            {
                "prob_a": [0.7, float("nan")],
                "prob_b": [0.3, 0.3],
            }
        )
        with pytest.raises(ValueError) as exc:
            integrity_checks(df)
        assert "NaN" in str(exc.value)

    def test_fails_on_sum_not_one(self):
        df = pd.DataFrame(
            {
                "prob_0": [0.6, 0.4],
                "prob_1": [0.5, 0.6],
            }
        )
        with pytest.raises(ValueError) as exc:
            integrity_checks(df)
        assert "sum" in str(exc.value)

    def test_fails_on_predicted_class_mismatch(self):
        df = pd.DataFrame(
            {
                "prob_0": [0.2, 0.8],
                "prob_1": [0.8, 0.2],
                "predicted_class": [0, 0],
            }
        )
        with pytest.raises(ValueError) as exc:
            integrity_checks(df)
        assert "predicted_class mismatch" in str(exc.value)

    def test_fails_on_predicted_label_mismatch(self):
        df = pd.DataFrame(
            {
                "prob_cat": [0.9, 0.1],
                "prob_dog": [0.1, 0.9],
                "predicted_label": ["dog", "dog"],
            }
        )
        with pytest.raises(ValueError) as exc:
            integrity_checks(df, prob_prefix="prob_")
        assert "predicted_label mismatch" in str(exc.value)

    def test_requires_probability_columns(self):
        df = pd.DataFrame({"predicted_class": [0, 1]})
        with pytest.raises(ValueError) as exc:
            integrity_checks(df)
        assert "no probability columns" in str(exc.value)


class TestGenerateTrustSummary:
    def test_summary_structure(self):
        metrics_summary = {"accuracy": 0.9}
        coverage_table = pd.DataFrame([
            {"group": "all", "coverage": 0.88},
            {"group": "minority", "coverage": 0.85},
        ])
        calibration_metrics = {"ece": 0.03}
        abstention_metrics = {"abstain_rate": 0.1}

        summary = generate_trust_summary(
            metrics_summary=metrics_summary,
            coverage_table=coverage_table,
            calibration_metrics=calibration_metrics,
            abstention_metrics=abstention_metrics,
        )

        assert summary["metrics_summary"] == metrics_summary
        assert summary["coverage_table"] == coverage_table.to_dict(orient="records")
        assert summary["calibration_metrics"] == calibration_metrics
        assert summary["abstention_metrics"] == abstention_metrics
