#!/usr/bin/env python3
"""
CLI for the Feature/Model registry.

Examples:
    foodspec-registry --registry registry.json --query-feature I_1742
    foodspec-registry --registry registry.json --query-protocol EdibleOil_Classification_v1 --version 0.1.0
"""

from __future__ import annotations

import argparse
import sys

from foodspec.registry import FeatureModelRegistry


def main(argv=None):
    parser = argparse.ArgumentParser(description="Query the FoodSpec registry.")
    parser.add_argument("--registry", required=True, help="Path to registry JSON.")
    parser.add_argument("--query-feature", help="Feature name to search for.")
    parser.add_argument("--query-protocol", help="Protocol name to search for.")
    parser.add_argument("--version", help="Protocol version filter.")
    args = parser.parse_args(argv)

    reg = FeatureModelRegistry(args.registry)
    if args.query_feature:
        entries = reg.query_by_feature(args.query_feature)
    elif args.query_protocol:
        entries = reg.query_by_protocol(args.query_protocol, args.version)
    else:
        print("Specify --query-feature or --query-protocol", file=sys.stderr)
        return 1

    for e in entries:
        print(f"Protocol {e.protocol_name} v{e.protocol_version} | model {e.model_id} | metrics {e.metrics}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
