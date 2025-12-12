"""
Environment check utility for FoodSpec.
Reports presence of key dependencies and optional extras (GUI/Web).
"""
from __future__ import annotations

import importlib
import sys


def check_env() -> str:
    msgs = []
    msgs.append(f"Python: {sys.version.split()[0]}")
    core = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "h5py"]
    gui = ["PyQt5"]
    web = ["fastapi", "uvicorn"]
    for mod in core:
        try:
            importlib.import_module(mod)
            msgs.append(f"{mod}: OK")
        except ImportError:
            msgs.append(f"{mod}: MISSING")
    gui_available = all(importlib.util.find_spec(m) for m in gui)
    web_available = all(importlib.util.find_spec(m) for m in web)
    msgs.append(f"GUI available: {'yes' if gui_available else 'no'}")
    msgs.append(f"Web API available: {'yes' if web_available else 'no'}")
    return "\n".join(msgs)


def main():
    print(check_env())


if __name__ == "__main__":  # pragma: no cover
    main()
