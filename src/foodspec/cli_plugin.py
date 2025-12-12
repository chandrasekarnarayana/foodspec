#!/usr/bin/env python3
"""
Plugin manager CLI for FoodSpec.

Examples:
    foodspec-plugin list
    foodspec-plugin install my_foodspec_plugin
"""
from __future__ import annotations

import argparse
import sys

from foodspec.plugin import PluginManager, install_plugin


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage FoodSpec plugins.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("list")
    install_p = sub.add_parser("install")
    install_p.add_argument("module", help="Module path to install/load.")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        pm = PluginManager()
        pm.discover()
        print("Protocols:")
        for p in pm.protocols:
            print(f"- {p.name} ({p.module})")
        print("Vendor loaders:")
        for k, v in pm.vendor_loaders.items():
            print(f"- {k} ({v.module})")
        print("Harmonization:")
        for k, v in pm.harmonization.items():
            print(f"- {k} ({v.module})")
        return 0
    elif args.cmd == "install":
        ok = install_plugin(args.module)
        if ok:
            print(f"Installed/loaded {args.module}")
            return 0
        print(f"Failed to install {args.module}", file=sys.stderr)
        return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
