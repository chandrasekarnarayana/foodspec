# Installation

```bash
pip install foodspec
pip install 'foodspec[gui]'    # GUI extras
pip install 'foodspec[web]'    # Web API extras
```

See README for platform notes and docs/cli_help.md for CLI flags.
# Installation (101)

This page walks you through installing FoodSpec and verifying your environment so you can run your first protocol.

## Requirements (and why)
- Python: 3.10–3.12 (tested matrix) – for dependency compatibility.
- OS: Windows, macOS, or Linux.
- RAM: ≥8 GB recommended for moderate datasets; more for large HSI cubes (HSI segmentation and harmonization can be memory-heavy).

## Install FoodSpec
Choose the extra that matches your workflow:
- Core (CLI, no GUI/web):  
  ```bash
  pip install foodspec
  ```
- With GUI (PyQt):  
  ```bash
  pip install "foodspec[gui]"
  ```
- With web API (FastAPI):  
  ```bash
  pip install "foodspec[web]"
  ```
- Dev/docs extras (optional): see `pyproject.toml` for `[dev]` and `[doc]`.

> Tip: Use a fresh virtual environment (`python -m venv .venv && source .venv/bin/activate` on macOS/Linux, `.venv\\Scripts\\activate` on Windows).

## Check your environment
Verify what’s available:
```bash
foodspec-run-protocol --check-env
# or
foodspec-predict --check-env
```
Expected output (abbreviated):
```
Python: 3.11
Core deps: OK
GUI available (PyQt): yes/no
Web API available (FastAPI): yes/no
```
Use this to confirm you installed the right extras before running GUI or web components.

## If you see error X, do Y
- **Missing PyQt / GUI unavailable**: Install GUI extra (`pip install "foodspec[gui]"`) or your platform’s Qt runtime.
- **Missing FastAPI/Uvicorn**: Install web extra (`pip install "foodspec[web]"`).
- **Protocol version error**: Your protocol’s `min_foodspec_version` exceeds the installed version; upgrade: `pip install --upgrade foodspec`.
- **Permission issues on Windows**: Run the shell as Administrator or install into a user venv.

If issues persist, capture the full error and open an issue with your command, OS, Python version, and a small data sample if possible.
