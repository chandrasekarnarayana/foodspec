# Installation

```bash
pip install foodspec
```

See README for platform notes and docs/cli_help.md for CLI flags.
# Installation (101)

This page walks you through installing FoodSpec and verifying your environment so you can run your first protocol.

## Requirements (and why)
- Python: 3.10–3.12 (tested matrix) – for dependency compatibility.
- OS: Windows, macOS, or Linux.
- RAM: ≥8 GB recommended for moderate datasets; more for large HSI cubes (HSI segmentation and harmonization can be memory-heavy).

## Install FoodSpec
FoodSpec provides a CLI-first workflow. Install the core package:
```bash
pip install foodspec
```
Optional dev/docs extras: see `pyproject.toml` for `[dev]`.

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
```
Use this to confirm you installed the right extras.

## If you see error X, do Y
- **Protocol version error**: Your protocol’s `min_foodspec_version` exceeds the installed version; upgrade: `pip install --upgrade foodspec`.
- **Permission issues on Windows**: Run the shell as Administrator or install into a user venv.
If issues persist, capture the full error and open an issue with your command, OS, Python version, and a small data sample if possible.

If issues persist, capture the full error and open an issue with your command, OS, Python version, and a small data sample if possible.
