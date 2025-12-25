import pandas as pd

from foodspec.protocol import ProtocolConfig, ProtocolRunner


def test_protocol_engine_runs_toy():
    # Toy protocol: no preprocess, simple rq_analysis on ratios present
    proto = {
        "name": "toy_protocol",
        "steps": [
            {
                "type": "rq_analysis",
                "params": {
                    "oil_col": "oil_type",
                    "matrix_col": "matrix",
                    "heating_col": "heating_stage",
                    "ratios": [{"name": "1742/2720", "numerator": "I_1742", "denominator": "I_2720"}],
                },
            },
            {"type": "output", "params": {"output_dir": "protocol_runs_test"}},
        ],
    }
    cfg = ProtocolConfig.from_dict(proto)
    runner = ProtocolRunner(cfg)
    df = pd.DataFrame(
        {
            "oil_type": ["A", "A", "B", "B"],
            "matrix": ["oil"] * 4,
            "heating_stage": [0, 1, 0, 1],
            "I_1742": [10, 9, 6, 5],
            "I_2720": [5, 5, 5, 5],
        }
    )
    res = runner.run([df])
    assert "stability_summary" in res.tables
    assert res.report
