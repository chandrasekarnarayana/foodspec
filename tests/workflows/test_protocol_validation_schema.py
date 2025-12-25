import pandas as pd

from foodspec.protocol import ProtocolConfig, validate_protocol


def test_protocol_validation_schema_checks():
    cfg = ProtocolConfig.from_dict(
        {
            "name": "test_proto",
            "version": "0.0.1",
            "steps": [{"type": "rq_analysis"}],
            "expected_columns": {"oil_col": "oil_type"},
            "min_foodspec_version": "0.0.0",
        }
    )
    df = pd.DataFrame({"oil_type": ["A", "B"]})
    diag = validate_protocol(cfg, df)
    assert not diag["errors"]
    assert "foodspec" in " ".join(diag["warnings"])
