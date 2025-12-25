"""
Example protocol plugin for FoodSpec.
Registers a trivial protocol template.
"""
from foodspec.protocol import ProtocolConfig


def get_plugins():
    proto = {
        "name": "Example_Plugin_Protocol",
        "version": "0.0.1",
        "steps": [
            {"type": "preprocess", "params": {"baseline_method": "als", "normalization": "reference"}},
            {"type": "rq_analysis", "params": {"oil_col": "oil_type", "matrix_col": "matrix"}},
            {"type": "output", "params": {"output_dir": "plugin_runs"}},
        ],
    }
    return {
        "protocols": [ProtocolConfig.from_dict(proto)],
        "vendor_loaders": {},
        "harmonization": {},
    }


def plugin_main():
    return get_plugins()
